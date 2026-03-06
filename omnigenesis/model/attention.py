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
        kv_state = torch.zeros(
            batch,
            self.heads,
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=acc_dtype,
        )
        k_state = torch.zeros(batch, self.heads, self.head_dim, device=x.device, dtype=acc_dtype)
        out_steps = []

        for t in range(n_tokens):
            kt = k[:, t].to(acc_dtype)
            vt = v[:, t].to(acc_dtype)
            qt = q[:, t].to(acc_dtype)

            kv_state = kv_state + torch.einsum("bhd,bhm->bhdm", kt, vt)
            k_state = k_state + kt

            denom_t = torch.einsum("bhd,bhd->bh", qt, k_state).clamp_min(1e-6)
            out_t = torch.einsum("bhd,bhdm->bhm", qt, kv_state) / denom_t.unsqueeze(-1)
            out_steps.append(out_t.to(q.dtype))

        out = torch.stack(out_steps, dim=1)
        return self.out(out.reshape(batch, n_tokens, dim))
