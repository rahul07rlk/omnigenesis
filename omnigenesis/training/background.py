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


def _make_loader(dataset, train_cfg: TrainConfig, device: str) -> DataLoader:
    kwargs = {
        "batch_size": train_cfg.batch_size,
        "pin_memory": train_cfg.pin_memory and (device == "cuda"),
        "num_workers": train_cfg.num_workers,
    }
    if train_cfg.num_workers > 0:
        kwargs["persistent_workers"] = train_cfg.persistent_workers
        kwargs["prefetch_factor"] = train_cfg.prefetch_factor
    return DataLoader(dataset, **kwargs)


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
) -> None:
    dataset = ResumableStreamingDataset(
        tokenizer,
        seq_len=train_cfg.seq_len,
        skip_seqs=start_seq if start_data_state is None else 0,
        resume_state=start_data_state,
        dataset_name=data_cfg.dataset_name,
        dataset_config=data_cfg.dataset_config,
        dataset_split=data_cfg.dataset_split,
        streaming=data_cfg.streaming,
        text_field=data_cfg.text_field,
        max_examples=data_cfg.max_examples,
        max_chars_per_example=data_cfg.max_chars_per_example,
    )
    loader = _make_loader(dataset, train_cfg, device)
    loader_iter = iter(loader)
    accum = train_cfg.grad_accum_steps

    step = start_step
    seq_count = start_seq
    batch_count = 0

    model.train()
    optimizer.zero_grad()
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
                dataset_state=dataset.get_resume_state(),
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
                dataset_state=dataset.get_resume_state(),
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
                dataset_state=dataset.get_resume_state(),
            )
            return
        except Exception as exc:
            print(f"[WARN] Data loading error at step {step}: {exc}", flush=True)
            traceback.print_exc()
            optimizer.zero_grad()
            save_checkpoint(
                model,
                optimizer,
                scaler,
                step,
                seq_count,
                dataset_state=dataset.get_resume_state(),
            )
            time.sleep(1.0)
            loader = _make_loader(dataset, train_cfg, device)
            loader_iter = iter(loader)
            continue

        try:
            use_non_blocking = device == "cuda"
            inputs = inputs.to(device, non_blocking=use_non_blocking)
            targets = targets.to(device, non_blocking=use_non_blocking)
        except Exception as exc:
            print(f"[WARN] Device transfer error at step {step}: {exc}", flush=True)
            traceback.print_exc()
            optimizer.zero_grad()
            continue

        with model_lock:
            try:
                with amp_context_factory():
                    out = model(inputs)
                    loss = model.total_loss(out, targets) / accum

                scaler.scale(loss).backward()
                batch_count += 1

                if batch_count % accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    step += 1

                    if step % train_cfg.log_every == 0:
                        print(
                            f"\n[step {step:05d}] loss={loss.item() * accum:.4f}"
                            f"  deep={out['n_deep']}  iters={out['n_iters']}",
                            flush=True,
                        )

            except Exception as exc:
                print(f"[WARN] Training error at step {step}: {exc}", flush=True)
                traceback.print_exc()
                optimizer.zero_grad()

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
                dataset_state=dataset.get_resume_state(),
            )
