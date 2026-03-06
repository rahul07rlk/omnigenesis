import os
import sys
import threading
import traceback
from typing import Optional

import torch
import torch.nn as nn

from ..concurrency import model_lock, shutdown_event, wait_for_training_thread
from ..config import InferenceConfig


def _sample_next_token(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    cfg: InferenceConfig,
) -> torch.Tensor:
    # Repetition penalty to reduce loops on short, weakly-trained models.
    if cfg.repetition_penalty > 1.0 and input_ids.numel() > 0:
        for b in range(logits.size(0)):
            seen = torch.unique(input_ids[b])
            logits[b, seen] = logits[b, seen] / cfg.repetition_penalty

    if not cfg.do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / cfg.temperature

    if cfg.top_k > 0 and cfg.top_k < logits.size(-1):
        topk_vals, _ = torch.topk(logits, cfg.top_k, dim=-1)
        kth = topk_vals[:, -1].unsqueeze(-1)
        logits = torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)

    if cfg.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        remove = cum_probs > cfg.top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False

        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(1, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def interactive_inference_loop(
    model: nn.Module,
    tokenizer,
    device: str,
    train_thread: Optional[threading.Thread] = None,
    inference_cfg: Optional[InferenceConfig] = None,
) -> None:
    inference_cfg = inference_cfg or InferenceConfig()

    print("\n" + "=" * 70)
    print("          OmniGenesis AGI is TRAINING in the background!")
    print("   Type any prompt + Enter. Type 'exit' or Ctrl+C to stop.")
    print("=" * 70 + "\n")

    while True:
        try:
            prompt = input("\033[1;36mYou: \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            prompt = "exit"

        if prompt.lower() in {"exit", "quit"}:
            print("\nSignalling training thread to stop and save...")
            shutdown_event.set()
            if not wait_for_training_thread(train_thread):
                os._exit(1)
            sys.exit(0)

        if not prompt:
            continue

        try:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            print("\033[1;32mAGI: \033[0m", end="", flush=True)
            for _ in range(inference_cfg.max_new_tokens):
                context = input_ids[:, -inference_cfg.max_context_tokens :]
                with model_lock:
                    prev_mode = model.training
                    model.eval()
                    try:
                        out = model(context)
                    finally:
                        model.train(prev_mode)
                next_tok = _sample_next_token(out["logits"][:, -1, :], input_ids, inference_cfg)
                input_ids = torch.cat([input_ids, next_tok], dim=-1)
                print(tokenizer.decode(next_tok[0]), end="", flush=True)
                if next_tok.item() == tokenizer.eos_token_id:
                    break
            print("\n")
        except Exception as exc:
            print(f"\nInference error: {exc}")
            traceback.print_exc()
