# OmniGenesis

[![CI](https://github.com/rahku/Omni_genesis/actions/workflows/ci.yml/badge.svg)](https://github.com/rahku/Omni_genesis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

Modular experimental training + interactive inference project built with PyTorch.

This repository runs a background training loop while serving interactive text generation in the foreground.  
The codebase has been split into focused modules to support easier maintenance and upgrades.

This project is open source and released under the terms of the [MIT License](LICENSE), and is free to use in personal, academic, and commercial settings subject to those terms.

## Highlights

- Modular package layout (`omnigenesis/`) instead of one large script.
- Concurrent training and inference with thread-safe model access.
- Resumable token-chunk dataset pipeline (streaming or non-streaming).
- Checkpoint save/load with optimizer and AMP scaler state.
- Weight tying preserved across checkpoint resume.
- Mixed precision support with safe CPU fallback.

## Repository Structure

```text
omni_genesis.py                      # Compatibility launcher (entrypoint)
requirements.txt
requirements-dev.txt
README.md
LICENSE
CODE_OF_CONDUCT.md
CONTRIBUTING.md
pytest.ini
pyproject.toml
.pre-commit-config.yaml
.github/workflows/ci.yml
omnigenesis.yaml                     # System profiles + runtime config
tests/
  test_model_smoke.py
  test_data_pipeline.py
omnigenesis/
  __init__.py
  app.py                             # Main startup orchestration
  config.py                          # Hyperparameters/config object
  concurrency.py                     # Shared lock/event/thread helpers
  data/
    streaming_dataset.py             # ResumableStreamingDataset
  model/
    rope.py                          # RoPE helpers
    attention.py                     # LinearAttention
    expert.py                        # DomainExpert
    dispatcher.py                    # MortonDispatcher
    novelty.py                       # NoveltyBuffer
    moe.py                           # UnifiedMoE
    reasoning.py                     # ReasoningLoop
    agi.py                           # OmniGenesisAGI
  inference/
    interactive.py                   # Interactive prompt loop
  training/
    checkpointing.py                 # save_checkpoint/load_checkpoint
    background.py                    # Background training thread loop
```

## Requirements

- Python 3.10+ recommended
- `pip` up to date
- GPU optional (CUDA-enabled PyTorch if you want GPU training)

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install PyTorch first from the official selector, then run:

```bash
python -m pip install -r requirements.txt --no-deps
```

## Testing and Quality

Run tests:

```bash
pytest
```

Run linter:

```bash
ruff check .
```

Optional pre-commit setup:

```bash
python -m pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
```

## Quick Start

From the repository root:

```bash
python omni_genesis.py
```

Google Colab (T4) recommended startup:

```bash
%cd /content
!git clone <your-repo-url>
%cd /content/<your-repo-folder>
!pip install -r requirements.txt
import os
os.environ["OMNI_PROFILE"] = "t4_colab"
!python omni_genesis.py
```

What happens at startup:

1. Device is selected (`cuda` if available, else `cpu`).
2. GPT-2 tokenizer is loaded.
3. `OmniGenesisAGI` model is initialized.
4. Checkpoint is loaded if present (`omnigenesis_ckpt.pt`).
5. Background training thread starts.
6. Interactive CLI loop starts in the main thread.

Exit behavior:

- Type `exit` or `quit`, or press `Ctrl+C`.
- App signals shutdown, waits for training thread to checkpoint, then exits.

## Checkpointing and Resume

Checkpoint file:

- `omnigenesis_ckpt.pt` in project root (default)

Saved state includes:

- model weights
- optimizer state
- AMP scaler state
- global step + sequence counters
- resumable dataset state (buffer/progress)

Resume behavior:

- If checkpoint exists, training resumes from saved state.
- Weight tying (`head.weight` and `embed.weight`) is explicitly restored after load.

## Configuration

System config is loaded from:

- [`omnigenesis.yaml`](omnigenesis.yaml)

Code reads this file at startup and builds:

- `AGIConfig` (model)
- `TrainConfig` (optimizer/loop/sequence)
- `DataConfig` (dataset + split)
- `InferenceConfig` (generation limits)

Important knobs:

- `dim`, `heads` for model size/attention shape
- `experts` for MoE capacity
- `max_reason_steps`, `reason_threshold` for reasoning loop
- novelty buffer sizing and threshold settings
- Morton dispatch options (`use_morton`, projection params)
- `seq_len`, `batch_size`, `grad_accum_steps`
- `max_steps` for bounded training runs
- dataset name/config/split + `max_examples` cap
- inference `max_new_tokens` + `max_context_tokens`

### Profile-based scaling

The YAML file supports `small`, `balanced`, `t4_colab`, and `large` profiles plus `auto` selection by VRAM.

Default `auto` behavior:

- <=4.5GB VRAM -> `small`
- <=10GB VRAM -> `balanced`
- <=16.5GB VRAM -> `t4_colab`
- above that -> `large`
- CPU-only -> `small`

GTX 1650 defaults (`small`):

- `dim=192`, `experts=4`, `max_reason_steps=2`
- `seq_len=64`, `batch_size=1`, `grad_accum_steps=16`
- default dataset: `daily_dialog` (English dialog), split `train`

Google Colab T4 defaults (`t4_colab`):

- `dim=320`, `heads=8`, `experts=8`
- `seq_len=128`, `batch_size=2`, `grad_accum_steps=8`
- dataset uses `daily_dialog` (English dialog), non-streaming
- inference uses sampling (`temperature`, `top_k`, `top_p`, repetition penalty)

### Runtime overrides (without code edits)

Use environment variables only to choose profile/file:

```bash
$env:OMNI_PROFILE="t4_colab"              # or small / balanced / large / auto
$env:OMNI_CONFIG_PATH="C:\path\to\omnigenesis.yaml"
python omni_genesis.py
```

For system-specific tuning, edit `omnigenesis.yaml` only.

## Data Pipeline

The dataset implementation is in [`omnigenesis/data/streaming_dataset.py`](omnigenesis/data/streaming_dataset.py).

- Uses Hugging Face `datasets` (streaming or regular mode)
- Produces fixed-length next-token chunks
- Tracks internal cursor/buffer state for resume

Current source:

- `daily_dialog`, split `train` by default
- configurable in `omnigenesis.yaml` (`profiles.*.data`)
- supports plain text, prompt/response pairs, and message-list chat schemas
- if dataset load fails (for example script-based datasets blocked by your `datasets` version), it auto-falls back to an internal English chat corpus so training continues
- you can force this local fallback by setting `dataset_name: builtin_english_chat`
- if `DataLoader` multiprocessing is blocked on Windows, training automatically retries with `num_workers=0`

## Training/Inference Concurrency Model

- A single global `model_lock` serializes model forward/backward/step and inference forward.
- Background training loop:
  - loads micro-batches from streaming dataset
  - applies AMP context (`autocast("cuda")` on GPU, `nullcontext()` on CPU)
  - accumulates gradients and steps optimizer periodically
  - saves checkpoints outside lock to reduce inference blocking
- Inference loop:
  - runs under `torch.no_grad()`
  - temporarily switches model to eval for each generation step
  - restores previous train/eval mode in `finally`
  - truncates context to a configurable window (`max_context_tokens`)

## Development Notes

- Main orchestration logic is in [`omnigenesis/app.py`](omnigenesis/app.py).
- `omni_genesis.py` remains as a stable launcher path.
- Import boundaries are split by responsibility to minimize coupling.

Recommended maintenance workflow:

1. Implement changes in the smallest relevant module.
2. Keep cross-module imports one-directional (app -> subsystems).
3. Add/adjust tests when behavior changes.
4. Run static checks and smoke runs before commit.

## Troubleshooting

`ModuleNotFoundError: No module named 'torch'`

- Install dependencies from `requirements.txt`.

Hugging Face dataset/network errors

- Verify internet access and that dataset endpoints are reachable.
- Retry startup; streaming errors are retried in training loop.

OOM on small GPUs

- Lower model dimensions/expert count.
- Reduce sequence length or generation length.
- Prefer smaller batch size and/or larger accumulation interval.

## Code of Conduct

This project follows the Contributor Covenant.

- See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## Contributing

Please read:

- [CONTRIBUTING.md](CONTRIBUTING.md)

Pull requests should keep modules focused, preserve thread safety, and include
clear reasoning for training-loop behavior changes.

## License

This project is licensed under the MIT License.

- See [LICENSE](LICENSE)
- Open-source use is permitted, including modification, redistribution, and commercial use, provided the copyright and license notice are retained.
