import torch
from torch.amp import GradScaler

from omnigenesis.config import AGIConfig, DataConfig, InferenceConfig, TrainConfig
from omnigenesis.inference.interactive import _sample_next_token
from omnigenesis.model.agi import OmniGenesisAGI
from omnigenesis.training.checkpointing import load_checkpoint, save_checkpoint


def _tiny_model():
    cfg = AGIConfig(
        {
            "vocab_size": 1024,
            "dim": 64,
            "heads": 4,
            "experts": 2,
            "max_reason_steps": 1,
            "novelty_buf_size": 32,
            "novelty_sketch_dim": 16,
            "morton_proj_dim": 4,
            "morton_bits": 4,
        }
    )
    model = OmniGenesisAGI(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler("cuda", enabled=False)
    return model, optimizer, scaler


def test_sample_next_token_handles_nan_logits():
    cfg = InferenceConfig(
        {
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        }
    )
    logits = torch.full((1, 16), float("nan"))
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    tok = _sample_next_token(logits, input_ids, cfg)
    assert tok.shape == (1, 1)
    assert tok.dtype == torch.long
    assert 0 <= tok.item() < logits.size(-1)


def test_load_checkpoint_unreadable_file_falls_back_cleanly(tmp_path):
    model, optimizer, scaler = _tiny_model()
    bad_path = tmp_path / "corrupt.pt"
    bad_path.write_bytes(b"not a torch checkpoint")

    step, seq_count, data_state = load_checkpoint(
        model,
        optimizer,
        scaler,
        filename=str(bad_path),
    )
    assert step == 0
    assert seq_count == 0
    assert data_state is None


def test_save_checkpoint_creates_parent_directory(tmp_path):
    model, optimizer, scaler = _tiny_model()
    ckpt_path = tmp_path / "nested" / "deeper" / "ckpt.pt"
    save_checkpoint(
        model,
        optimizer,
        scaler,
        step=3,
        seq_count=42,
        filename=str(ckpt_path),
        dataset_state={"buffer": [1, 2], "seqs_emitted": 2},
    )
    assert ckpt_path.exists()


def test_total_loss_supports_label_smoothing():
    model, _, _ = _tiny_model()
    model.train()
    inputs = torch.randint(0, model.cfg.vocab_size, (2, 12))
    targets = torch.randint(0, model.cfg.vocab_size, (2, 12))
    out = model(inputs)
    loss = model.total_loss(out, targets, label_smoothing=0.1)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_train_and_data_config_parse_overfit_controls():
    train_cfg = TrainConfig(
        {
            "label_smoothing": 0.08,
            "val_every_steps": 100,
            "val_batches": 4,
            "eval_num_workers": 1,
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 0.002,
            "lr_scheduler_patience": 2,
            "lr_scheduler_factor": 0.6,
            "min_lr": 1e-5,
            "save_best_checkpoint": True,
        }
    )
    data_cfg = DataConfig(
        {
            "dataset_name": "builtin_english_chat",
            "dataset_split": "train",
            "eval_split": "validation",
            "eval_streaming": False,
            "eval_max_examples": 64,
        }
    )
    assert train_cfg.label_smoothing == 0.08
    assert train_cfg.val_every_steps == 100
    assert train_cfg.early_stopping_patience == 3
    assert train_cfg.lr_scheduler_patience == 2
    assert data_cfg.eval_split == "validation"
    assert data_cfg.eval_streaming is False
    assert data_cfg.eval_max_examples == 64


def test_checkpoint_roundtrip_with_scheduler_state(tmp_path):
    model, optimizer, scaler = _tiny_model()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )
    ckpt_path = tmp_path / "with_sched.pt"
    save_checkpoint(
        model,
        optimizer,
        scaler,
        step=7,
        seq_count=99,
        filename=str(ckpt_path),
        scheduler=scheduler,
    )

    model2, optimizer2, scaler2 = _tiny_model()
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2,
        mode="min",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )
    step, seq_count, _ = load_checkpoint(
        model2,
        optimizer2,
        scaler2,
        filename=str(ckpt_path),
        scheduler=scheduler2,
    )
    assert step == 7
    assert seq_count == 99


def test_load_checkpoint_with_legacy_ffn_index_keys(tmp_path):
    model, optimizer, scaler = _tiny_model()
    state = model.state_dict()
    legacy_state = {}
    for key, value in state.items():
        if ".ffn.2." in key:
            legacy_state[key.replace(".ffn.2.", ".ffn.3.")] = value.clone()
        else:
            legacy_state[key] = value.clone()

    ckpt_path = tmp_path / "legacy_keys.pt"
    payload = {
        "step": 5,
        "seq_count": 11,
        "model_state_dict": legacy_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    torch.save(payload, ckpt_path)

    model2, optimizer2, scaler2 = _tiny_model()
    step, seq_count, _ = load_checkpoint(model2, optimizer2, scaler2, filename=str(ckpt_path))
    assert step == 5
    assert seq_count == 11
