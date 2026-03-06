import os
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.amp import GradScaler


def _atomic_save_with_retry(payload: dict, filename: str, retries: int = 4, delay_s: float = 0.35) -> bool:
    target = Path(filename)
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        tmp_name = (
            f"{target.name}.tmp-{os.getpid()}-{threading.get_ident()}-{int(time.time() * 1000)}"
        )
        tmp_path = target.with_name(tmp_name)
        try:
            torch.save(payload, str(tmp_path))
            os.replace(str(tmp_path), str(target))
            return True
        except Exception as exc:
            last_error = exc
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            if attempt < retries:
                time.sleep(delay_s * (attempt + 1))

    ts = time.strftime("%H:%M:%S")
    fallback_name = f"{target.stem}.failedsave-{int(time.time())}{target.suffix or '.pt'}"
    fallback = target.with_name(fallback_name)
    try:
        torch.save(payload, str(fallback))
        print(
            f"\n[{ts}] [WARN] Could not replace '{target.name}' ({last_error}). "
            f"Saved fallback checkpoint: {fallback.name}",
            flush=True,
        )
        return True
    except Exception as fallback_exc:
        print(
            f"\n[{ts}] [ERROR] Checkpoint save failed for '{target.name}' "
            f"and fallback save failed: {fallback_exc}",
            flush=True,
        )
        return False


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
    ok = _atomic_save_with_retry(payload, filename)
    if not ok:
        return
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
        except Exception as exc:
            print(
                f"[WARN] Checkpoint exists but is incompatible with current config: {exc}",
                flush=True,
            )
            print("[INFO] Starting from scratch. Delete checkpoint if this is expected.", flush=True)
            return 0, 0, None
        scaler_state = ckpt.get("scaler_state_dict")
        if scaler_state is not None:
            try:
                scaler.load_state_dict(scaler_state)
            except Exception as exc:
                print(
                    f"[WARN] Could not restore GradScaler state: {exc}. Continuing with fresh scaler.",
                    flush=True,
                )
        model.head.weight = model.embed.weight
        step, seq_count = ckpt["step"], ckpt.get("seq_count", 0)
        dataset_state = ckpt.get("dataset_state")
        print(f"[OK] Resumed: step={step} | sequences={seq_count}", flush=True)
        return step, seq_count, dataset_state
    print("[INFO] No checkpoint found - starting from scratch.", flush=True)
    return 0, 0, None
