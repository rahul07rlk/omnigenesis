import os
import time
from typing import Optional, Tuple

import torch
from torch.amp import GradScaler


def save_checkpoint(
    model,
    optimizer,
    scaler: GradScaler,
    step: int,
    seq_count: int,
    filename: str = "omnigenesis_ckpt.pt",
    dataset_state: Optional[dict] = None,
) -> None:
    payload = {
        "step": step,
        "seq_count": seq_count,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    if dataset_state is not None:
        payload["dataset_state"] = dataset_state
    torch.save(payload, filename)
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] [OK] Checkpoint saved: step={step} | sequences={seq_count}", flush=True)


def load_checkpoint(
    model,
    optimizer,
    scaler: GradScaler,
    filename: str = "omnigenesis_ckpt.pt",
) -> Tuple[int, int, Optional[dict]]:
    if os.path.exists(filename):
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            ckpt = torch.load(filename, map_location=map_location, weights_only=True)
        except TypeError:
            ckpt = torch.load(filename, map_location=map_location)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception as exc:
            print(
                f"[WARN] Checkpoint exists but is incompatible with current config: {exc}",
                flush=True,
            )
            print("[INFO] Starting from scratch. Delete checkpoint if this is expected.", flush=True)
            return 0, 0, None
        model.head.weight = model.embed.weight
        step, seq_count = ckpt["step"], ckpt.get("seq_count", 0)
        dataset_state = ckpt.get("dataset_state")
        print(f"[OK] Resumed: step={step} | sequences={seq_count}", flush=True)
        return step, seq_count, dataset_state
    print("[INFO] No checkpoint found - starting from scratch.", flush=True)
    return 0, 0, None
