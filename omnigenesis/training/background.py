import time
import traceback
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..concurrency import model_lock, shutdown_event
from ..config import DataConfig, TrainConfig
from ..data.streaming_dataset import ResumableStreamingDataset
from .checkpointing import save_checkpoint


def _make_loader(dataset, train_cfg: TrainConfig, device: str, num_workers: Optional[int] = None) -> DataLoader:
    worker_count = train_cfg.num_workers if num_workers is None else max(0, int(num_workers))
    kwargs = {
        "batch_size": train_cfg.batch_size,
        "pin_memory": train_cfg.pin_memory and (device == "cuda"),
        "num_workers": worker_count,
    }
    if worker_count > 0:
        kwargs["persistent_workers"] = train_cfg.persistent_workers
        kwargs["prefetch_factor"] = train_cfg.prefetch_factor
    return DataLoader(dataset, **kwargs)


def _build_loader_iter(dataset, train_cfg: TrainConfig, device: str, requested_workers: int):
    attempts = [max(0, int(requested_workers))]
    if attempts[0] > 0:
        attempts.append(0)
    last_exc: Optional[Exception] = None
    for workers in attempts:
        try:
            loader = _make_loader(dataset, train_cfg, device, num_workers=workers)
            return loader, iter(loader), workers
        except Exception as exc:
            last_exc = exc
            print(
                f"[WARN] DataLoader init failed with num_workers={workers}: {exc}",
                flush=True,
            )
    raise RuntimeError("Unable to initialize DataLoader.") from last_exc


def background_training_loop(
    model: nn.Module,
    optimizer,
    scaler: GradScaler,
    tokenizer,
    device: str,
    start_step: int,
    start_seq: int,
    start_data_state: Optional[dict],
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    ckpt_path: str,
) -> None:
    effective_workers = train_cfg.num_workers
    if effective_workers > 0 and (start_seq > 0 or start_data_state is not None):
        print(
            "[WARN] Resume requested with num_workers>0; switching to num_workers=0 for exact resume.",
            flush=True,
        )
        effective_workers = 0
    resume_state = start_data_state if effective_workers == 0 else None
    skip_seqs = start_seq if resume_state is None else 0

    dataset = ResumableStreamingDataset(
        tokenizer,
        seq_len=train_cfg.seq_len,
        skip_seqs=skip_seqs,
        resume_state=resume_state,
        dataset_name=data_cfg.dataset_name,
        dataset_config=data_cfg.dataset_config,
        dataset_split=data_cfg.dataset_split,
        streaming=data_cfg.streaming,
        text_field=data_cfg.text_field,
        max_examples=data_cfg.max_examples,
        max_chars_per_example=data_cfg.max_chars_per_example,
        english_only=data_cfg.english_only,
        min_english_ratio=data_cfg.min_english_ratio,
        chat_messages_field=data_cfg.chat_messages_field,
        chat_role_field=data_cfg.chat_role_field,
        chat_content_field=data_cfg.chat_content_field,
        chat_role_fallback_fields=data_cfg.chat_role_fallback_fields,
        chat_content_fallback_fields=data_cfg.chat_content_fallback_fields,
        prompt_field=data_cfg.prompt_field,
        response_field=data_cfg.response_field,
        prompt_fallback_fields=data_cfg.prompt_fallback_fields,
        response_fallback_fields=data_cfg.response_fallback_fields,
        max_turns_per_example=data_cfg.max_turns_per_example,
    )
    loader, loader_iter, effective_workers = _build_loader_iter(
        dataset,
        train_cfg,
        device,
        effective_workers,
    )
    accum = train_cfg.grad_accum_steps
    track_dataset_state = effective_workers == 0
    get_dataset_state = dataset.get_resume_state if track_dataset_state else (lambda: None)

    step = start_step
    seq_count = start_seq
    batch_count = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    print("[TRAIN] Background training started...", flush=True)
    amp_context_factory = (lambda: autocast("cuda")) if device == "cuda" else (lambda: nullcontext())

    while True:
        if train_cfg.max_steps > 0 and step >= train_cfg.max_steps:
            print(f"\n[Training] Reached max_steps={train_cfg.max_steps} - saving checkpoint...", flush=True)
            save_checkpoint(
                model,
                optimizer,
                scaler,
                step,
                seq_count,
                filename=ckpt_path,
                dataset_state=get_dataset_state(),
            )
            return

        if shutdown_event.is_set():
            print("\n[Training] Shutdown received - saving checkpoint...", flush=True)
            save_checkpoint(
                model,
                optimizer,
                scaler,
                step,
                seq_count,
                filename=ckpt_path,
                dataset_state=get_dataset_state(),
            )
            return

        try:
            inputs, targets = next(loader_iter)
        except StopIteration:
            print("\n[Training] Data stream ended - saving checkpoint...", flush=True)
            save_checkpoint(
                model,
                optimizer,
                scaler,
                step,
                seq_count,
                filename=ckpt_path,
                dataset_state=get_dataset_state(),
            )
            return
        except Exception as exc:
            print(f"[WARN] Data loading error at step {step}: {exc}", flush=True)
            traceback.print_exc()
            optimizer.zero_grad(set_to_none=True)
            save_checkpoint(
                model,
                optimizer,
                scaler,
                step,
                seq_count,
                filename=ckpt_path,
                dataset_state=get_dataset_state(),
            )
            time.sleep(1.0)
            loader, loader_iter, effective_workers = _build_loader_iter(
                dataset,
                train_cfg,
                device,
                effective_workers,
            )
            if effective_workers == 0 and not track_dataset_state:
                track_dataset_state = True
                get_dataset_state = dataset.get_resume_state
            continue

        try:
            use_non_blocking = device == "cuda"
            inputs = inputs.to(device, non_blocking=use_non_blocking)
            targets = targets.to(device, non_blocking=use_non_blocking)
        except Exception as exc:
            print(f"[WARN] Device transfer error at step {step}: {exc}", flush=True)
            traceback.print_exc()
            optimizer.zero_grad(set_to_none=True)
            continue

        with model_lock:
            try:
                with amp_context_factory():
                    out = model(inputs)
                    raw_loss = model.total_loss(out, targets)
                if not torch.isfinite(raw_loss):
                    print(
                        f"[WARN] Non-finite loss at step {step}; skipping update and clearing gradients.",
                        flush=True,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss = raw_loss / accum

                scaler.scale(loss).backward()
                batch_count += 1

                if batch_count % accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1

                    if step % train_cfg.log_every == 0:
                        print(
                            f"\n[step {step:05d}] loss={raw_loss.item():.4f}"
                            f"  deep={out['n_deep']}  iters={out['n_iters']}",
                            flush=True,
                        )

            except Exception as exc:
                print(f"[WARN] Training error at step {step}: {exc}", flush=True)
                traceback.print_exc()
                optimizer.zero_grad(set_to_none=True)

        seq_count += inputs.size(0)

        if (
            step > 0
            and step % train_cfg.ckpt_every == 0
            and batch_count % accum == 0
        ):
            save_checkpoint(
                model,
                optimizer,
                scaler,
                step,
                seq_count,
                filename=ckpt_path,
                dataset_state=get_dataset_state(),
            )
