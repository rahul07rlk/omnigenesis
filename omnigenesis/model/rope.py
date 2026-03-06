import torch


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    _, n_tokens, _, d_head = x.shape
    half = d_head // 2
    pos = torch.arange(n_tokens, device=x.device, dtype=x.dtype).unsqueeze(-1)
    dim_idx = torch.arange(half, device=x.device, dtype=x.dtype)
    theta = 1.0 / (10000.0 ** (dim_idx / half))
    angles = pos * theta
    sin = angles.sin()[None, :, None, :]
    cos = angles.cos()[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
