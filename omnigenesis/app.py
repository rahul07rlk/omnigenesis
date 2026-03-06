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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RUN] Running on: {device.upper()}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = OmniGenesisAGI(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
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
    print(
        "[RUN] dataset: "
        f"{data_cfg.dataset_name}/{data_cfg.dataset_config} "
        f"split={data_cfg.dataset_split} streaming={data_cfg.streaming}",
        flush=True,
    )
    print(
        "[RUN] decoding: "
        f"sample={inference_cfg.do_sample} temp={inference_cfg.temperature} "
        f"top_k={inference_cfg.top_k} top_p={inference_cfg.top_p} "
        f"rep_pen={inference_cfg.repetition_penalty}",
        flush=True,
    )

    step, seq_count, data_state = load_checkpoint(model, optimizer, scaler)

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
