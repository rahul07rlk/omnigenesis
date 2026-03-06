import torch
import torch.nn as nn


class MortonDispatcher(nn.Module):
    """
    Locality-aware token ordering using Morton/Z-order codes.

    Sorting selected token indices by Morton code improves locality patterns
    before expert execution.
    """

    def __init__(self, in_dim: int, proj_dim: int, bits: int):
        super().__init__()
        self.bits = bits
        self.proj_dim = proj_dim
        self.levels = 1 << bits
        proj = torch.randn(in_dim, proj_dim)
        q, _ = torch.linalg.qr(proj)
        self.register_buffer("proj", q[:, :proj_dim].contiguous())
        d_idx = torch.arange(proj_dim)
        b_idx = torch.arange(bits)
        shifts = d_idx.unsqueeze(1) + (b_idx.unsqueeze(0) * proj_dim)
        self.register_buffer("shifts", shifts.long())

    def morton_codes(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x.float() @ self.proj.float()
        x_scaled = (torch.tanh(x_proj) + 1.0) * 0.5
        q = (x_scaled.clamp(0, 1 - 1e-6) * (self.levels - 1)).long()
        q_exp = q.unsqueeze(-1)
        b_idx = torch.arange(self.bits, device=x.device).view(1, 1, -1)
        bit_vals = (q_exp >> b_idx) & 1
        contrib = bit_vals << self.shifts.unsqueeze(0)
        return contrib.reshape(x.size(0), -1).sum(dim=-1)

    def sort(self, flat: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        if pos.numel() <= 32:
            return pos
        codes = self.morton_codes(flat[pos].detach())
        _, order = torch.sort(codes)
        return pos[order]
