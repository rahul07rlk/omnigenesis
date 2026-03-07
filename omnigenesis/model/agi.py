from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AGIConfig
from .moe import UnifiedMoE
from .novelty import NoveltyBuffer
from .reasoning import ReasoningLoop


class OmniGenesisAGI(nn.Module):
    """
    Main orchestration model:
    embedding -> novelty gate -> (shallow or deep MoE+reasoning path) -> logits.
    """

    def __init__(self, cfg: AGIConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.moe = UnifiedMoE(cfg)
        self.proj_reason = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim),
            nn.LayerNorm(cfg.dim),
        )
        self.reasoning = ReasoningLoop(cfg.dim, cfg.max_reason_steps, cfg.reason_threshold)
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.novelty_buf = NoveltyBuffer(cfg.dim, cfg.novelty_buf_size, cfg.novelty_sketch_dim)

    def forward(self, input_ids: torch.Tensor) -> Dict:
        batch, _ = input_ids.shape
        x = self.emb_drop(self.embed(input_ids))
        z_seq = x.mean(dim=1)

        novelty = self.novelty_buf.novelty_score(z_seq.detach())
        deep_mask = novelty > self.cfg.novelty_threshold

        if self.training:
            self.novelty_buf.update(z_seq.detach())

        shallow = self.norm(x)

        if deep_mask.any():
            idx = deep_mask.nonzero(as_tuple=False).squeeze(1)
            x_d = x[idx]
            x_moe, z_loss, aux_loss = self.moe(x_d)
            z_reason = self.proj_reason(x_moe.mean(dim=1))
            ctx = z_seq[idx]
            z_out, n_iters, confidence_sel = self.reasoning(z_reason, ctx)
            deep_out = self.norm(x_moe + z_out.unsqueeze(1))

            deep_full = torch.zeros_like(shallow)
            deep_full = deep_full.index_copy(0, idx, deep_out.to(dtype=deep_full.dtype))
            mask3d = deep_mask.view(batch, 1, 1).expand_as(shallow)
            final = torch.where(mask3d, deep_full, shallow)
            confidence = x.new_zeros(batch)
            confidence = confidence.index_copy(0, idx, confidence_sel.to(dtype=confidence.dtype))
        else:
            final = shallow
            z_loss = x.new_zeros(())
            aux_loss = x.new_zeros(())
            n_iters = 0
            confidence = x.new_zeros(batch)

        return {
            "logits": self.head(final),
            "z_loss": z_loss,
            "aux_loss": aux_loss,
            "confidence": confidence,
            "n_iters": n_iters,
            "n_deep": int(deep_mask.sum().item()),
        }

    def ce_loss(self, logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
        kwargs = {"ignore_index": -1}
        if label_smoothing > 0:
            kwargs["label_smoothing"] = float(label_smoothing)
        try:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                **kwargs,
            )
        except TypeError:
            kwargs.pop("label_smoothing", None)
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                **kwargs,
            )

    def total_loss(self, out: dict, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
        ce = self.ce_loss(out["logits"], targets, label_smoothing=label_smoothing)
        return ce + 0.01 * out["aux_loss"] + 0.001 * out["z_loss"]
