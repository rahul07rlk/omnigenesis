import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = dim // heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.scale = float(self.head_dim ** -0.25)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_tokens, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, n_tokens, self.heads, self.head_dim)
        k = k.view(batch, n_tokens, self.heads, self.head_dim)
        v = v.view(batch, n_tokens, self.heads, self.head_dim)
        q, k = apply_rope(q), apply_rope(k)
        q = F.elu(q * self.scale) + 1.0
        k = F.elu(k * self.scale) + 1.0

        acc_dtype = torch.float32 if q.dtype in {torch.float16, torch.bfloat16} else q.dtype
        q_acc = q.to(acc_dtype)
        k_acc = k.to(acc_dtype)
        v_acc = v.to(acc_dtype)

        # Prefix accumulators over time for linear attention.
        kv_prefix = torch.cumsum(k_acc.unsqueeze(-1) * v_acc.unsqueeze(-2), dim=1)
        k_prefix = torch.cumsum(k_acc, dim=1)

        numer = torch.einsum("bthd,bthdm->bthm", q_acc, kv_prefix)
        denom = torch.einsum("bthd,bthd->bth", q_acc, k_prefix).clamp_min(1e-6)
        out = (numer / denom.unsqueeze(-1)).to(q.dtype)
        return self.out(out.reshape(batch, n_tokens, dim))
