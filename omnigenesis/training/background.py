import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..concurrency import model_lock, shutdown_event
from ..config import DataConfig, TrainConfig
from ..data.streaming_dataset import ResumableStreamingDataset
from .checkpointing import save_checkpoint

MAX_CONSECUTIVE_ERRORS = 25
BASE_BACKOFF_SECONDS = 0.25
MAX_BACKOFF_SECONDS = 5.0


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


def _is_builtin_corpus(dataset_obj) -> bool:
    return dataset_obj.__class__.__name__ == "_BuiltinEnglishCorpus"


def _build_eval_stream(
    tokenizer,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    device: str,
    train_dataset,
):
    if train_cfg.val_every_steps <= 0:
        return None, None, None
    if not data_cfg.eval_split:
        print("[INFO] Validation disabled because data.eval_split is empty.", flush=True)
        return None, None, None

    eval_streaming = data_cfg.streaming if data_cfg.eval_streaming is None else data_cfg.eval_streaming
    eval_max_examples = data_cfg.eval_max_examples if data_cfg.eval_max_examples > 0 else data_cfg.max_examples

    try:
        eval_dataset = ResumableStreamingDataset(
            tokenizer,
            seq_len=train_cfg.seq_len,
            skip_seqs=0,
            resume_state=None,
            dataset_name=data_cfg.dataset_name,
            dataset_config=data_cfg.dataset_config,
            dataset_split=data_cfg.eval_split,
            streaming=eval_streaming,
            text_field=data_cfg.text_field,
            max_examples=eval_max_examples,
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
        _, eval_loader_iter, eval_workers = _build_loader_iter(
            eval_dataset,
            train_cfg,
            device,
            train_cfg.eval_num_workers,
        )
    except Exception as exc:
        print(f"[WARN] Validation dataset setup failed: {exc}", flush=True)
        traceback.print_exc()
        return None, None, None

    train_uses_builtin = _is_builtin_corpus(train_dataset.dataset)
    eval_uses_builtin = _is_builtin_corpus(eval_dataset.dataset)
    dataset_name = data_cfg.dataset_name.strip().lower()
    explicitly_builtin = dataset_name in {"builtin_english_chat", "internal_english_chat"}
    if eval_uses_builtin and not train_uses_builtin and not explicitly_builtin:
        print(
            "[WARN] Validation split fell back to built-in corpus while training split did not. "
            "Disabling validation to avoid misleading early-stopping decisions.",
            flush=True,
        )
        return None, None, None

    print(
        f"[INFO] Validation enabled: split={data_cfg.eval_split} "
        f"batches={train_cfg.val_batches} every={train_cfg.val_every_steps} steps",
        flush=True,
    )
    return eval_dataset, eval_loader_iter, eval_workers


def _run_validation(
    model: nn.Module,
    train_cfg: TrainConfig,
    device: str,
    eval_dataset,
    eval_loader_iter,
    eval_workers: int,
    amp_context_factory,
):
    if eval_dataset is None or eval_loader_iter is None:
        return None, eval_loader_iter, eval_workers

    total_losses = []
    ce_losses = []

    for _ in range(train_cfg.val_batches):
        try:
            inputs, targets = next(eval_loader_iter)
        except StopIteration:
            _, eval_loader_iter, eval_workers = _build_loader_iter(
                eval_dataset,
                train_cfg,
                device,
                eval_workers,
            )
            try:
                inputs, targets = next(eval_loader_iter)
            except StopIteration:
                break
        except Exception as exc:
            print(f"[WARN] Validation data error: {exc}", flush=True)
            traceback.print_exc()
            return None, eval_loader_iter, eval_workers

        try:
            use_non_blocking = device == "cuda"
            inputs = inputs.to(device, non_blocking=use_non_blocking)
            targets = targets.to(device, non_blocking=use_non_blocking)
        except Exception as exc:
            print(f"[WARN] Validation device transfer error: {exc}", flush=True)
            traceback.print_exc()
            return None, eval_loader_iter, eval_workers

        with model_lock:
            prev_mode = model.training
            model.eval()
            try:
                with torch.no_grad():
                    with amp_context_factory():
                        out = model(inputs)
                        total = model.total_loss(out, targets, label_smoothing=0.0)
                        ce = model.ce_loss(out["logits"], targets, label_smoothing=0.0)
            finally:
                model.train(prev_mode)

        if not (torch.isfinite(total) and torch.isfinite(ce)):
            print("[WARN] Non-finite validation loss encountered; skipping validation result.", flush=True)
            return None, eval_loader_iter, eval_workers

        total_losses.append(float(total.item()))
        ce_losses.append(float(ce.item()))

    if not total_losses:
        print("[WARN] Validation produced no batches.", flush=True)
        return None, eval_loader_iter, eval_workers

    avg_total = sum(total_losses) / len(total_losses)
    avg_ce = sum(ce_losses) / len(ce_losses)
    return (avg_total, avg_ce), eval_loader_iter, eval_workers


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
    scheduler=None,
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
    _, loader_iter, effective_workers = _build_loader_iter(
        dataset,
        train_cfg,
        device,
        effective_workers,
    )
    accum = train_cfg.grad_accum_steps
    track_dataset_state = effective_workers == 0
    get_dataset_state = dataset.get_resume_state if track_dataset_state else (lambda: None)
    amp_context_factory = (lambda: autocast("cuda")) if device == "cuda" else (lambda: nullcontext())

    eval_dataset, eval_loader_iter, eval_workers = _build_eval_stream(
        tokenizer,
        train_cfg,
        data_cfg,
        device,
        dataset,
    )

    ckpt_target = Path(ckpt_path)
    suffix = ckpt_target.suffix or ".pt"
    best_ckpt_path = str(ckpt_target.with_name(f"{ckpt_target.stem}.best{suffix}"))

    step = start_step
    seq_count = start_seq
    batch_count = 0
    consecutive_errors = 0
    best_val_ce = float("inf")
    evals_without_improvement = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    print("[TRAIN] Background training started...", flush=True)

    def _save_progress() -> None:
        save_checkpoint(
            model,
            optimizer,
            scaler,
            step,
            seq_count,
            filename=ckpt_path,
            dataset_state=get_dataset_state(),
            scheduler=scheduler,
        )

    def _record_error(stage: str, exc: Exception) -> bool:
        nonlocal consecutive_errors
        consecutive_errors += 1
        print(
            f"[WARN] {stage} error at step {step}: {exc} "
            f"(consecutive={consecutive_errors}/{MAX_CONSECUTIVE_ERRORS})",
            flush=True,
        )
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            print(
                "[ERROR] Too many consecutive training errors; saving checkpoint and stopping thread.",
                flush=True,
            )
            _save_progress()
            return False
        if shutdown_event.is_set():
            print(
                "[Training] Shutdown requested while handling an error - saving checkpoint.",
                flush=True,
            )
            _save_progress()
            return False
        sleep_s = min(
            MAX_BACKOFF_SECONDS,
            BASE_BACKOFF_SECONDS * (2 ** min(consecutive_errors - 1, 5)),
        )
        time.sleep(sleep_s)
        return True

    while True:
        if train_cfg.max_steps > 0 and step >= train_cfg.max_steps:
            print(f"\n[Training] Reached max_steps={train_cfg.max_steps} - saving checkpoint...", flush=True)
            _save_progress()
            return

        if shutdown_event.is_set():
            print("\n[Training] Shutdown received - saving checkpoint...", flush=True)
            _save_progress()
            return

        try:
            inputs, targets = next(loader_iter)
        except StopIteration:
            # Unlimited runs should continue cycling finite datasets unless max_examples is explicitly bounded.
            if train_cfg.max_steps == 0 and data_cfg.max_examples == 0:
                print("\n[Training] Data stream ended - restarting data iterator...", flush=True)
                _, loader_iter, effective_workers = _build_loader_iter(
                    dataset,
                    train_cfg,
                    device,
                    effective_workers,
                )
                if effective_workers == 0 and not track_dataset_state:
                    track_dataset_state = True
                    get_dataset_state = dataset.get_resume_state
                consecutive_errors = 0
                continue
            print("\n[Training] Data stream ended - saving checkpoint...", flush=True)
            _save_progress()
            return
        except Exception as exc:
            traceback.print_exc()
            optimizer.zero_grad(set_to_none=True)
            if not _record_error("Data loading", exc):
                return
            try:
                _, loader_iter, effective_workers = _build_loader_iter(
                    dataset,
                    train_cfg,
                    device,
                    effective_workers,
                )
                if effective_workers == 0 and not track_dataset_state:
                    track_dataset_state = True
                    get_dataset_state = dataset.get_resume_state
            except Exception as reload_exc:
                traceback.print_exc()
                if not _record_error("DataLoader reinit", reload_exc):
                    return
            continue

        try:
            use_non_blocking = device == "cuda"
            inputs = inputs.to(device, non_blocking=use_non_blocking)
            targets = targets.to(device, non_blocking=use_non_blocking)
        except Exception as exc:
            traceback.print_exc()
            optimizer.zero_grad(set_to_none=True)
            if not _record_error("Device transfer", exc):
                return
            continue

        train_exc: Optional[Exception] = None
        non_finite_loss = False
        batch_ok = False
        step_advanced = False
        with model_lock:
            try:
                with amp_context_factory():
                    out = model(inputs)
                    raw_loss = model.total_loss(
                        out,
                        targets,
                        label_smoothing=train_cfg.label_smoothing,
                    )
                if not torch.isfinite(raw_loss):
                    non_finite_loss = True
                    optimizer.zero_grad(set_to_none=True)
                else:
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
                        step_advanced = True

                        if step % train_cfg.log_every == 0:
                            print(
                                f"\n[step {step:05d}] loss={raw_loss.item():.4f}"
                                f"  deep={out['n_deep']}  iters={out['n_iters']}",
                                flush=True,
                            )
                    batch_ok = True
            except Exception as exc:
                traceback.print_exc()
                optimizer.zero_grad(set_to_none=True)
                train_exc = exc

        if train_exc is not None:
            if not _record_error("Training", train_exc):
                return
            continue
        if non_finite_loss:
            if not _record_error("Non-finite loss", RuntimeError("loss is not finite")):
                return
            continue
        if not batch_ok:
            if not _record_error("Training", RuntimeError("batch was not processed")):
                return
            continue

        consecutive_errors = 0
        seq_count += inputs.size(0)

        if (
            step_advanced
            and train_cfg.val_every_steps > 0
            and eval_dataset is not None
            and step % train_cfg.val_every_steps == 0
        ):
            val_result, eval_loader_iter, eval_workers = _run_validation(
                model,
                train_cfg,
                device,
                eval_dataset,
                eval_loader_iter,
                eval_workers,
                amp_context_factory,
            )
            if val_result is not None:
                val_total, val_ce = val_result
                if scheduler is not None:
                    try:
                        scheduler.step(val_ce)
                    except Exception as exc:
                        print(f"[WARN] Scheduler step failed: {exc}", flush=True)
                current_lr = optimizer.param_groups[0].get("lr", train_cfg.lr)
                print(
                    f"[val step {step:05d}] total={val_total:.4f} ce={val_ce:.4f} lr={current_lr:.6g}",
                    flush=True,
                )

                improved = val_ce < (best_val_ce - train_cfg.early_stopping_min_delta)
                if improved:
                    best_val_ce = val_ce
                    evals_without_improvement = 0
                    if train_cfg.save_best_checkpoint:
                        save_checkpoint(
                            model,
                            optimizer,
                            scaler,
                            step,
                            seq_count,
                            filename=best_ckpt_path,
                            dataset_state=get_dataset_state(),
                            scheduler=scheduler,
                        )
                        print(
                            f"[INFO] New best validation CE={best_val_ce:.4f}; "
                            f"saved best checkpoint: {best_ckpt_path}",
                            flush=True,
                        )
                else:
                    evals_without_improvement += 1
                    if (
                        train_cfg.early_stopping_patience > 0
                        and evals_without_improvement >= train_cfg.early_stopping_patience
                    ):
                        print(
                            "[Training] Early stopping triggered: "
                            f"no validation improvement for {evals_without_improvement} evals.",
                            flush=True,
                        )
                        _save_progress()
                        return

        if (
            step > 0
            and step % train_cfg.ckpt_every == 0
            and batch_count % accum == 0
        ):
            _save_progress()
