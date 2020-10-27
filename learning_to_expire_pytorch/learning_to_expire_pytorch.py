import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from collections import namedtuple

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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

    def forward(self, x, mem = None):
        n, h, scale, device = x.shape[1], self.heads, self.scale, x.device

        q = self.to_q(x)

        mem_len = 0
        if exists(mem):
            mem_len = mem.shape[1]
            x = torch.cat((mem, x), dim = 1)

        kv = self.to_kv(x).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        mask = torch.ones(dots.shape[-2:], device = device).triu_(mem_len + 1).bool()
        mask = rearrange(mask, 'i j -> () () i j')
        dots.masked_fill_(mask, float('-inf'))
        del mask

        attn = dots.softmax(dim = -1)

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
        num_memory_blocks,
        heads = 8):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CausalAttention(dim, heads = heads)),
                PreNorm(dim, FeedForward(dim)),
            ]))

        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x, mems = None):
        x = self.token_emb(x)

        hidden_states = []
        mems = default(mems, (None,) * self.depth)
        for (mem, (attn, ff)) in zip(mems, self.layers):
            hidden_states.append(x)

            x = attn(x, mem = mem) + x
            x = ff(x) + x

        return self.to_logits(x), hidden_states
