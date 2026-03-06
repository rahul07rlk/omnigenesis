import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AGIConfig
from .dispatcher import MortonDispatcher
from .expert import DomainExpert


class UnifiedMoE(nn.Module):
    """
    Top-2 routed Mixture-of-Experts block.

    Router computes token-expert probabilities (with optional Gumbel noise
    during training), dispatches to selected experts, and combines weighted
    expert outputs back to token positions.
    """

    def __init__(self, cfg: AGIConfig):
        super().__init__()
        self.E = cfg.experts
        self.router = nn.Linear(cfg.dim, cfg.experts)
        self.experts = nn.ModuleList([DomainExpert(cfg.dim, cfg.heads) for _ in range(cfg.experts)])
        self.dispatcher = (
            MortonDispatcher(cfg.dim, cfg.morton_proj_dim, cfg.morton_bits) if cfg.use_morton else None
        )

    def forward(self, x: torch.Tensor, tau: float = 1.0):
        batch, n_tokens, dim = x.shape
        total = batch * n_tokens
        flat = x.view(total, dim)

        logits = self.router(flat)
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2)

        if self.training:
            uniform = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            probs = F.softmax((logits + gumbel) / tau, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)

        top_p, top_i = torch.topk(probs, 2, dim=-1)
        top_p = top_p / (top_p.sum(dim=-1, keepdim=True) + 1e-12)

        out = torch.zeros(total, dim, dtype=flat.dtype, device=flat.device)

        for eid in range(self.E):
            sel0 = top_i[:, 0] == eid
            sel1 = top_i[:, 1] == eid
            sel = sel0 | sel1
            pos = sel.nonzero(as_tuple=False).squeeze(1)
            if pos.numel() == 0:
                continue
            if self.dispatcher is not None:
                pos = self.dispatcher.sort(flat, pos)
            tok = flat.index_select(0, pos)
            proc = self.experts[eid](tok.unsqueeze(1)).squeeze(1)
            w = (top_p[pos, 0] * sel0[pos].float()) + (top_p[pos, 1] * sel1[pos].float())
            weighted = proc * w.unsqueeze(-1)
            out = out.scatter_add(0, pos.unsqueeze(-1).expand_as(weighted), weighted)

        importance = probs.mean(dim=0)
        fi = F.one_hot(top_i, num_classes=self.E).float().sum(dim=1).mean(dim=0)
        aux_loss = float(self.E) * (importance * fi).sum()

        return out.view(batch, n_tokens, dim), z_loss, aux_loss
