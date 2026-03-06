import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoveltyBuffer(nn.Module):
    def __init__(self, dim: int, size: int, sketch_dim: int):
        super().__init__()
        self.size = size
        rand_proj = torch.randint(0, 2, (dim, sketch_dim)).float() * 2 - 1
        rand_proj = rand_proj / math.sqrt(sketch_dim)
        self.register_buffer("R", rand_proj)
        self.register_buffer("buffer", torch.zeros(size, sketch_dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long))

    def _sketch(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float() @ self.R.float(), dim=-1)

    def novelty_score(self, x: torch.Tensor) -> torch.Tensor:
        if self.filled.item() == 0:
            return torch.ones(x.size(0), device=x.device)
        sx = self._sketch(x)
        active = self.buffer[: int(self.filled.item())]
        sim = sx @ active.T
        return 1.0 - sim.max(dim=1).values

    def update(self, x: torch.Tensor):
        sx = self._sketch(x.detach())
        n = sx.size(0)
        ptr = int(self.ptr.item())
        first = min(n, self.size - ptr)
        if first > 0:
            self.buffer[ptr : ptr + first] = sx[:first]
        rem = n - first
        if rem > 0:
            self.buffer[0:rem] = sx[first : first + rem]
        self.ptr.fill_((ptr + n) % self.size)
        self.filled.fill_(min(int(self.filled.item()) + n, self.size))
