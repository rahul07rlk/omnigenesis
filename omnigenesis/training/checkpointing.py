import os
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.amp import GradScaler


def _remap_model_state_for_compat(model, state_dict: dict) -> Optional[dict]:
    target_keys = set(model.state_dict().keys())
    remapped = dict(state_dict)
    changed = False
    for key, value in state_dict.items():
        if ".ffn.3." in key:
            candidate = key.replace(".ffn.3.", ".ffn.2.")
            if candidate in target_keys and candidate not in remapped:
                remapped[candidate] = value
                changed = True
        if ".ffn.2." in key:
            candidate = key.replace(".ffn.2.", ".ffn.3.")
            if candidate in target_keys and candidate not in remapped:
                remapped[candidate] = value
                changed = True
    return remapped if changed else None


def _atomic_save_with_retry(payload: dict, filename: str, retries: int = 4, delay_s: float = 0.35) -> bool:
    target = Path(filename)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(
            f"[ERROR] Could not create checkpoint directory '{target.parent}': {exc}",
            flush=True,
        )
        return False
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
    scheduler=None,
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
    if scheduler is not None:
        try:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        except Exception as exc:
            print(f"[WARN] Could not serialize scheduler state: {exc}", flush=True)
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
    scheduler=None,
) -> Tuple[int, int, Optional[dict]]:
    if os.path.exists(filename):
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            try:
                ckpt = torch.load(filename, map_location=map_location, weights_only=True)
            except TypeError:
                ckpt = torch.load(filename, map_location=map_location)
        except Exception as exc:
            print(
                f"[WARN] Could not read checkpoint '{filename}': {exc}",
                flush=True,
            )
            print("[INFO] Starting from scratch due to unreadable checkpoint.", flush=True)
            return 0, 0, None
        model_state = ckpt.get("model_state_dict")
        if not isinstance(model_state, dict):
            print("[WARN] Checkpoint missing model_state_dict. Starting from scratch.", flush=True)
            return 0, 0, None
        try:
            model.load_state_dict(model_state)
        except Exception as exc:
            remapped_state = _remap_model_state_for_compat(model, model_state)
            if remapped_state is None:
                print(
                    f"[WARN] Checkpoint exists but is incompatible with current config: {exc}",
                    flush=True,
                )
                print("[INFO] Starting from scratch. Delete checkpoint if this is expected.", flush=True)
                return 0, 0, None
            try:
                model.load_state_dict(remapped_state)
                print("[INFO] Loaded checkpoint with compatibility key remapping.", flush=True)
            except Exception as remap_exc:
                print(
                    f"[WARN] Checkpoint remap attempt failed: {remap_exc}",
                    flush=True,
                )
                print("[INFO] Starting from scratch. Delete checkpoint if this is expected.", flush=True)
                return 0, 0, None
        optimizer_state = ckpt.get("optimizer_state_dict")
        if optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
            except Exception as exc:
                print(
                    f"[WARN] Could not restore optimizer state: {exc}. Continuing with fresh optimizer.",
                    flush=True,
                )
        scaler_state = ckpt.get("scaler_state_dict")
        if scaler_state is not None:
            try:
                scaler.load_state_dict(scaler_state)
            except Exception as exc:
                print(
                    f"[WARN] Could not restore GradScaler state: {exc}. Continuing with fresh scaler.",
                    flush=True,
                )
        scheduler_state = ckpt.get("scheduler_state_dict")
        if scheduler is not None and scheduler_state is not None:
            try:
                scheduler.load_state_dict(scheduler_state)
            except Exception as exc:
                print(
                    f"[WARN] Could not restore scheduler state: {exc}. Continuing with fresh scheduler.",
                    flush=True,
                )
        model.head.weight = model.embed.weight
        step, seq_count = ckpt["step"], ckpt.get("seq_count", 0)
        dataset_state = ckpt.get("dataset_state")
        print(f"[OK] Resumed: step={step} | sequences={seq_count}", flush=True)
        return step, seq_count, dataset_state
    print("[INFO] No checkpoint found - starting from scratch.", flush=True)
    return 0, 0, None
