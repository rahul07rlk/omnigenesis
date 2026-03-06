import torch
import torch.nn as nn

from omnigenesis.config import AGIConfig
from omnigenesis.model.agi import OmniGenesisAGI
from omnigenesis.model.moe import UnifiedMoE


def _tiny_cfg() -> AGIConfig:
    return AGIConfig(
        {
            "vocab_size": 1024,
            "dim": 64,
            "heads": 4,
            "experts": 2,
            "max_reason_steps": 1,
            "novelty_buf_size": 64,
            "novelty_sketch_dim": 32,
            "morton_proj_dim": 4,
            "morton_bits": 4,
        }
    )


def test_model_forward_smoke():
    cfg = _tiny_cfg()
    model = OmniGenesisAGI(cfg)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids)

    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert isinstance(out["n_deep"], int)
    assert 0 <= out["n_deep"] <= 2


def test_total_loss_scalar_and_finite():
    cfg = _tiny_cfg()
    model = OmniGenesisAGI(cfg)
    model.train()

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids)
    loss = model.total_loss(out, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_forward_handles_confidence_dtype_mismatch():
    cfg = AGIConfig(
        {
            "vocab_size": 1024,
            "dim": 64,
            "heads": 4,
            "experts": 2,
            "max_reason_steps": 1,
            "reason_threshold": 0.0,
            "novelty_threshold": -1.0,
            "novelty_buf_size": 64,
            "novelty_sketch_dim": 32,
            "morton_proj_dim": 4,
            "morton_bits": 4,
        }
    )
    model = OmniGenesisAGI(cfg)
    model.train()

    class _HalfConfidenceReasoning(nn.Module):
        def forward(self, z, ctx):
            return z, 1, torch.ones(z.size(0), device=z.device, dtype=torch.float16)

    model.reasoning = _HalfConfidenceReasoning()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(input_ids)

    assert out["confidence"].dtype == model.embed.weight.dtype
    assert out["confidence"].shape == (2,)


def test_moe_single_expert_routing_works():
    cfg = AGIConfig(
        {
            "vocab_size": 1024,
            "dim": 64,
            "heads": 4,
            "experts": 1,
            "max_reason_steps": 1,
            "novelty_buf_size": 64,
            "novelty_sketch_dim": 32,
            "morton_proj_dim": 4,
            "morton_bits": 4,
        }
    )
    moe = UnifiedMoE(cfg)
    x = torch.randn(2, 8, cfg.dim)
    y, z_loss, aux_loss = moe(x)

    assert y.shape == x.shape
    assert torch.isfinite(z_loss).item()
    assert torch.isfinite(aux_loss).item()
