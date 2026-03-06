import torch

from omnigenesis.config import AGIConfig
from omnigenesis.model.agi import OmniGenesisAGI


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
