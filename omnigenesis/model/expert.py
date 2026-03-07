import threading

import torch
import torch.nn as nn

from .attention import LinearAttention


class DomainExpert(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = LinearAttention(dim, heads)
        self.resid_drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.ffn_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.register_buffer("grad_momentum", torch.zeros(1))
        self.frozen = False
        self._tls = threading.local()

        def hook_fn(grad):
            self._on_grad()
            return grad

        for param in self.parameters():
            if param.requires_grad:
                param.register_hook(hook_fn)

    def _on_grad(self):
        if self.frozen:
            return
        if not hasattr(self._tls, "calls"):
            self._tls.calls = 0
        self._tls.calls += 1
        n_params = sum(1 for p in self.parameters() if p.requires_grad)
        if n_params == 0:
            return
        if self._tls.calls >= n_params:
            self._update_plasticity()
            self._tls.calls = 0

    def _update_plasticity(self):
        total, count = 0.0, 0
        for param in self.parameters():
            if param.grad is not None:
                total += param.grad.norm().item()
                count += 1
        if count > 0:
            self.grad_momentum.mul_(0.9).add_(0.1 * (total / count))
        if self.grad_momentum.item() < 1e-6:
            for param in self.parameters():
                param.requires_grad = False
            self.frozen = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen:
            x_detached = x.detach()
            with torch.no_grad():
                h = self.resid_drop(self.attn(self.norm1(x_detached)))
                delta = self.resid_drop(self.ffn_drop(self.ffn(self.norm2(x_detached + h))))
            return x + delta
        h = self.resid_drop(self.attn(self.norm1(x)))
        return x + self.resid_drop(self.ffn_drop(self.ffn(self.norm2(x + h))))
