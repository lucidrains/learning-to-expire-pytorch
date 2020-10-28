import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from collections import namedtuple

# constants

Memory = namedtuple('Memory', ['mems', 'elapsed_times'])

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def safe_cat(tensors, dim = -1):
    tensors = list(filter(exists, tensors))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim = dim)

def safe_add(tensor, n):
    if not exists(tensor):
        return None
    return tensor + n

# expire span logic

class ExpireSpan(nn.Module):
    def __init__(self, dim, max_mem_len, ramp_length):
        super().__init__()
        self.max_mem_len = max_mem_len
        self.ramp_length = ramp_length
        self.to_expiration = nn.Linear(dim, 1)
        nn.init.constant_(self.to_expiration.bias.data, val = -self.max_mem_len)

    def forward(self, mem, time, seq_len):
        exps = self.to_expiration(mem).squeeze(-1).sigmoid() * self.max_mem_len
        exps = rearrange(exps, 'b j -> b () () j')
        t = rearrange(time, 'j -> () j')
        r = F.pad(exps - t, (0, seq_len), value = 1.)
        mask = torch.clamp((r / self.ramp_length) + 1, min = 0., max = 1.)
        return exps, mask

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class CausalAttention(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mem = None, expire_mask = None):
        n, h, scale, device = x.shape[1], self.heads, self.scale, x.device

        q = self.to_q(x)

        mem_len = mem.shape[1] if exists(mem) else 0
        context = safe_cat((mem, x), dim = 1)

        kv = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        mask = torch.ones(dots.shape[-2:], device = device).triu_(mem_len + 1).bool()
        mask = rearrange(mask, 'i j -> () () i j')
        dots.masked_fill_(mask, float('-inf'))
        del mask

        attn = dots.softmax(dim = -1)

        if exists(expire_mask):
            attn  = attn * expire_mask

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class ExpireSpanTransformerXL(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        seq_len,
        heads = 8,
        num_memory_blocks = 10,
        expire_loss_coef = 1e-6,
        ramp_length = 128):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.depth = depth
        self.seq_len = seq_len
        self.max_mem_len = num_memory_blocks * seq_len

        self.expire_loss_coef = expire_loss_coef

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ExpireSpan(dim, self.max_mem_len, ramp_length),
                PreNorm(dim, CausalAttention(dim, heads = heads)),
                PreNorm(dim, FeedForward(dim)),
            ]))

        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x, memory = None):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)

        hidden_states = []
        mems_layers = memory.mems if exists(memory) else ((None,) * self.depth)
        times_layers = memory.elapsed_times if exists(memory) else ((None,) * self.depth)
        aux_loss = torch.tensor(0., requires_grad = True)

        for (mem, time, (expire_span, attn, ff)) in zip(mems_layers, times_layers, self.layers):
            hidden_states.append(x)

            exps, expire_mask = expire_span(mem, time, seq_len = n) if exists(mem) else (None, None)

            x = attn(x, mem = mem, expire_mask = expire_mask) + x
            x = ff(x) + x

            if exists(exps):
                aux_loss = aux_loss + (exps / self.seq_len).sum() * self.expire_loss_coef

        logits = self.to_logits(x)

        if self.seq_len == n:
            new_memories = map(lambda t: safe_cat(t, dim = 1), list(zip(mems_layers, hidden_states)))
            new_memories = map(lambda t: t[:, -self.max_mem_len:].detach(), new_memories)

            new_times = torch.arange(n - 1, -1, -1, device = device)
            new_elapsed_times = map(lambda t: safe_cat((safe_add(t, n), new_times), dim = 0), times_layers)
            new_elapsed_times = map(lambda t: t[-self.max_mem_len:], new_elapsed_times)

            memory = Memory(list(new_memories), list(new_elapsed_times))

        return logits, memory, aux_loss
