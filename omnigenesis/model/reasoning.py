import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ReasoningLoop(nn.Module):
    def __init__(self, dim: int, max_iters: int, threshold: float):
        super().__init__()
        self.max_iters = max_iters
        self.tau = threshold
        self.refine = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.conf_head = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.depth_emb = nn.Embedding(max_iters + 1, dim)

    def _step(self, z: torch.Tensor, ctx: torch.Tensor, n_tensor: torch.Tensor) -> torch.Tensor:
        depth = self.depth_emb(n_tensor)
        return self.norm(z + self.refine(torch.cat([z + depth, ctx], dim=-1)))

    def forward(self, z: torch.Tensor, ctx: torch.Tensor):
        batch = z.size(0)
        done = torch.zeros(batch, dtype=torch.bool, device=z.device)
        n = -1
        kappa = torch.zeros(batch, device=z.device)

        for n in range(self.max_iters):
            n_tensor = torch.full((batch,), n, device=z.device, dtype=torch.long)
            if self.training:
                z_new = checkpoint(self._step, z, ctx, n_tensor, use_reentrant=False)
            else:
                z_new = self._step(z, ctx, n_tensor)
            kappa = torch.sigmoid(self.conf_head(z_new)).squeeze(-1)
            newly_done = kappa >= self.tau
            z = torch.where(done.unsqueeze(-1), z, z_new)
            done = done | newly_done
            if done.all():
                break

        return z, n + 1, kappa
