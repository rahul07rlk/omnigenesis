import os
import sys
import threading

import torch
from torch.amp import GradScaler
from transformers import AutoTokenizer

from .concurrency import shutdown_event, wait_for_training_thread
from .config import active_profile, cfg, config_source_path, data_cfg, inference_cfg, train_cfg
from .inference.interactive import interactive_inference_loop
from .model.agi import OmniGenesisAGI
from .training.background import background_training_loop
from .training.checkpointing import load_checkpoint


def main():
    shutdown_event.clear()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RUN] Running on: {device.upper()}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # Disable GPT-2 max-length warnings during dataset tokenization; sequence windows are controlled by config.
    tokenizer.model_max_length = 1_000_000_000

    model = OmniGenesisAGI(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = None
    if train_cfg.lr_scheduler_patience > 0 and train_cfg.lr_scheduler_factor < 1.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=train_cfg.lr_scheduler_factor,
            patience=train_cfg.lr_scheduler_patience,
            min_lr=train_cfg.min_lr,
        )
    scaler = GradScaler("cuda", enabled=(device == "cuda"))

    print(
        "[RUN] profile: "
        f"name={active_profile}, dim={cfg.dim}, heads={cfg.heads}, experts={cfg.experts}, "
        f"seq_len={train_cfg.seq_len}, batch={train_cfg.batch_size}, "
        f"accum={train_cfg.grad_accum_steps}",
        flush=True,
    )
    print(f"[RUN] config: {config_source_path}", flush=True)
    dataset_cfg = data_cfg.dataset_config or "-"
    print(
        "[RUN] dataset: "
        f"{data_cfg.dataset_name}/{dataset_cfg} "
        f"split={data_cfg.dataset_split} streaming={data_cfg.streaming} "
        f"english_only={data_cfg.english_only}",
        flush=True,
    )
    print(
        "[RUN] decoding: "
        f"sample={inference_cfg.do_sample} temp={inference_cfg.temperature} "
        f"top_k={inference_cfg.top_k} top_p={inference_cfg.top_p} "
        f"rep_pen={inference_cfg.repetition_penalty}",
        flush=True,
    )
    print(
        "[RUN] regularization: "
        f"dropout={cfg.dropout} label_smoothing={train_cfg.label_smoothing} "
        f"val_every={train_cfg.val_every_steps} early_stop_patience={train_cfg.early_stopping_patience}",
        flush=True,
    )
    ckpt_path = os.getenv("OMNI_CKPT_PATH", f"omnigenesis_ckpt_{active_profile}.pt")
    print(f"[RUN] checkpoint: {ckpt_path}", flush=True)

    step, seq_count, data_state = load_checkpoint(
        model,
        optimizer,
        scaler,
        filename=ckpt_path,
        scheduler=scheduler,
    )

    train_thread = threading.Thread(
        target=background_training_loop,
        args=(
            model,
            optimizer,
            scaler,
            tokenizer,
            device,
            step,
            seq_count,
            data_state,
            train_cfg,
            data_cfg,
            ckpt_path,
            scheduler,
        ),
        name="TrainingThread",
    )
    train_thread.start()

    try:
        interactive_inference_loop(model, tokenizer, device, train_thread, inference_cfg)
    finally:
        shutdown_event.set()
        if not wait_for_training_thread(train_thread):
            os._exit(1)


def run():
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
