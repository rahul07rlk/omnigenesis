# OmniGenesis

[![CI](https://github.com/rahku/Omni_genesis/actions/workflows/ci.yml/badge.svg)](https://github.com/rahku/Omni_genesis/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

OmniGenesis is a modular PyTorch project for **concurrent language-model training and interactive chat inference** in one process.

The system is designed around adaptive compute:
- easy inputs use a shallow path
- novel inputs route through deeper MoE + reasoning
- training runs continuously in a background thread while you chat in the foreground

This repository is open source under the [MIT License](LICENSE).

## What Makes This Model Different

OmniGenesis combines several methods to push efficiency on limited and high-end GPUs:

1. **Novelty-gated depth**
Only sequences that look novel go through the expensive deep path.

2. **Top-2 Mixture of Experts (MoE)**
Tokens are routed to a small subset of experts instead of all experts.

3. **Morton (Z-order) dispatch**
Selected token indices are locality-sorted before expert execution for better memory access behavior.

4. **Linear attention with RoPE**
Experts use linear-time prefix attention (instead of quadratic full attention) plus rotary position encoding.

5. **Iterative reasoning loop**
A confidence head decides when to stop refining latent state, so compute can stop early.

6. **Concurrent train + infer runtime**
One lock-protected model supports safe updates and safe generation from separate threads.

## End-to-End System Flow

```text
                 +-----------------------------+
                 |  omnigenesis.yaml profiles |
                 +-------------+---------------+
                               |
                               v
                  +---------------------------+
                  | app.py startup orchestration |
                  +---------------------------+
                     | device/tokenizer/model
                     | checkpoint resume
                     v
         +----------------------+    +----------------------+
         | Training Thread      |    | Main Thread          |
         | background.py        |    | interactive.py       |
         +----------------------+    +----------------------+
         | stream/chunk dataset |    | prompt -> generate   |
         | forward/backward AMP |    | top-k/top-p sampling |
         | optimizer step       |    | repetition penalty   |
         | periodic checkpoint  |    |                      |
         +----------+-----------+    +-----------+----------+
                    \_____________________ __________________/
                                          v
                                  shared model + lock
```

## Model Flow (Inside `OmniGenesisAGI`)

Implementation: [`omnigenesis/model/agi.py`](omnigenesis/model/agi.py)

1. **Token embedding**
`input_ids -> embed -> x` where `x` is `[batch, tokens, dim]`.

2. **Sequence summary**
`z_seq = mean(x, dim=tokens)`.

3. **Novelty scoring**
`NoveltyBuffer.novelty_score(z_seq)` returns `1 - max_cosine_similarity` against a compact memory sketch.

4. **Depth decision**
`deep_mask = novelty > novelty_threshold`.

5. **Shallow path**
Always computed as `LayerNorm(x)`.

6. **Deep path (only for novel rows)**
`x_d -> UnifiedMoE -> reason projection -> ReasoningLoop -> residual merge + norm`.

7. **Selective merge**
Deep outputs are copied back only for deep rows, shallow outputs are used for others.

8. **Tied output head**
`head.weight = embed.weight` and logits are produced with tied weights.

### Loss Function

Defined in `OmniGenesisAGI.total_loss`:

```text
total_loss = cross_entropy + 0.01 * aux_loss + 0.001 * z_loss
```

- `cross_entropy`: next-token objective
- `aux_loss`: MoE load-balancing term
- `z_loss`: router stabilizer from `logsumexp(logits)^2`

## Core Technologies in Detail

### 1) Novelty Buffer

Implementation: [`omnigenesis/model/novelty.py`](omnigenesis/model/novelty.py)

- Uses a random sign projection matrix `R` to sketch embeddings into a low-dimensional normalized space.
- Keeps a ring buffer of sketches.
- Novelty score is inverse nearest similarity to memory.
- During training, sequence sketches are continuously added.

Why it matters:
- pushes expensive compute only when needed
- reduces average inference/training compute

### 2) Unified MoE (Top-2 Router)

Implementation: [`omnigenesis/model/moe.py`](omnigenesis/model/moe.py)

- Router maps each token to expert logits.
- Training can inject Gumbel noise before softmax for exploration.
- Top-2 experts (or top-1 if only one expert) are selected per token.
- Outputs from selected experts are combined by normalized routing weights.
- Includes auxiliary balancing loss and router regularization (`z_loss`).

Why it matters:
- more capacity than a dense block at similar compute budget
- expert specialization emerges over training

### 3) Morton Dispatcher

Implementation: [`omnigenesis/model/dispatcher.py`](omnigenesis/model/dispatcher.py)

- Projects token states into quantized coordinates.
- Interleaves coordinate bits into Morton codes.
- Sorts routed token indices by code before expert execution.

Why it matters:
- improves locality patterns for batched expert work
- helps keep routing overhead efficient

### 4) Domain Experts (Linear Attention + FFN)

Implementation: [`omnigenesis/model/expert.py`](omnigenesis/model/expert.py), [`omnigenesis/model/attention.py`](omnigenesis/model/attention.py)

- Each expert uses:
  - linear attention with RoPE
  - residual MLP block
  - layer norms
- Linear attention uses prefix accumulators instead of full attention matrix.
- Expert parameters can auto-freeze when gradient momentum remains near zero.

Why it matters:
- lower time/memory growth with longer sequences than standard full attention
- expert freezing can reduce unnecessary updates

### 5) Iterative Reasoning Loop

Implementation: [`omnigenesis/model/reasoning.py`](omnigenesis/model/reasoning.py)

- Refines latent state `z` for up to `max_reason_steps`.
- Confidence head predicts `kappa`; loop stops early when `kappa >= reason_threshold`.
- Uses activation checkpointing during training to save memory.

Why it matters:
- adaptive computation depth
- memory-efficient deep refinement

## Data Pipeline

Implementation: [`omnigenesis/data/streaming_dataset.py`](omnigenesis/data/streaming_dataset.py)

Features:
- Hugging Face dataset loading in streaming or non-streaming mode
- resilient fallback to built-in English chat corpus if dataset load fails
- schema-flexible text extraction:
  - plain text fields
  - chat messages arrays
  - prompt/response pairs
- optional English-like filtering (`english_only`, `min_english_ratio`)
- fixed-length token chunking for next-token prediction
- resumable internal state (`buffer`, emitted sequences, examples seen, skips)

Notes:
- `builtin_english_chat` is a tiny internal fallback corpus for robustness, not a replacement for large-scale data.
- For real quality gains, use large clean corpora (for example FineWeb sample configs already provided).

## Training Runtime

Implementation: [`omnigenesis/training/background.py`](omnigenesis/training/background.py)

- Background thread:
  - creates resumable dataset iterator
  - uses AMP autocast on CUDA
  - applies gradient accumulation + clipping
  - periodically checkpoints
- Uses a shared `model_lock` so train and infer do not race.
- Handles non-finite loss by skipping update and clearing grads.
- Automatically retries DataLoader with `num_workers=0` if multiprocessing fails.

## Checkpointing

Implementation: [`omnigenesis/training/checkpointing.py`](omnigenesis/training/checkpointing.py)

- Saves:
  - model state
  - optimizer state
  - GradScaler state
  - global step + sequence count
  - dataset resume state
- Uses atomic temp-file replace with retry.
- If replace fails, falls back to a `.failedsave-*` checkpoint file.
- Restores tied weights after load.

## Inference Runtime

Implementation: [`omnigenesis/inference/interactive.py`](omnigenesis/inference/interactive.py)

- Interactive CLI loop with configurable:
  - `max_new_tokens`
  - `max_context_tokens`
  - sampling/greedy
  - temperature, top-k, top-p
  - sign-aware repetition penalty
- Context is truncated to configured max window to bound cost.

## Configuration and Profiles

Config source: [`omnigenesis.yaml`](omnigenesis.yaml)

Profile selection:
- `runtime.active_profile: auto` (default)
- override with `OMNI_PROFILE`

Current profiles:
- `small` (4GB class)
- `balanced` (6-10GB class)
- `t4_colab` (T4 class)
- `large` (16-32GB class)
- `a100` (40GB+ class)

Default auto VRAM rules:
- `<= 4.5GB -> small`
- `<= 10GB -> balanced`
- `<= 16.5GB -> t4_colab`
- `<= 32GB -> large`
- `> 32GB -> a100`

Environment overrides:

```bash
# Windows PowerShell
$env:OMNI_PROFILE="a100"  # auto|small|balanced|t4_colab|large|a100
$env:OMNI_CONFIG_PATH="C:\path\to\omnigenesis.yaml"
$env:OMNI_CKPT_PATH="C:\path\to\checkpoint.pt"
python omni_genesis.py
```

## Quick Start

### 1) Install

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need a specific CUDA build, install PyTorch first from official wheels, then:

```bash
python -m pip install -r requirements.txt --no-deps
```

### 2) Run

```bash
python omni_genesis.py
```

Startup sequence:
1. detect device
2. load tokenizer
3. build model from profile config
4. resume checkpoint if present
5. start training thread
6. enter interactive chat loop

Exit:
- type `exit` or `quit`
- or `Ctrl+C`

### 3) Colab T4 Example

```python
%cd /content
!git clone <your-repo-url>
%cd /content/<your-repo-folder>
!pip install -r requirements.txt
import os
os.environ["OMNI_PROFILE"] = "t4_colab"
!python omni_genesis.py
```

### 4) A100 Example

```bash
export OMNI_PROFILE=a100
python omni_genesis.py
```

## Repository Structure

```text
omni_genesis.py                      # Launcher entrypoint
omnigenesis.yaml                     # Runtime profiles and defaults
omnigenesis/
  app.py                             # Main orchestration
  config.py                          # Config loading and profile resolve
  concurrency.py                     # Lock/event/thread helpers
  data/streaming_dataset.py          # Resumable dataset pipeline
  model/
    agi.py                           # Main model graph and total loss
    moe.py                           # Router + expert dispatch/merge
    dispatcher.py                    # Morton sorting
    expert.py                        # Expert block
    attention.py                     # Linear attention + RoPE
    novelty.py                       # Novelty sketch memory
    reasoning.py                     # Iterative confidence loop
  training/
    background.py                    # Training thread
    checkpointing.py                 # Save/load with retries
  inference/interactive.py           # CLI generation loop
tests/
  test_model_smoke.py
  test_data_pipeline.py
```

## Testing and Quality

Run tests:

```bash
pytest
```

Run lint:

```bash
ruff check .
```

Optional pre-commit:

```bash
python -m pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
```

## Troubleshooting

### `Dataset scripts are no longer supported`

Your `datasets` version blocked script-style dataset loaders.  
OmniGenesis will auto-fallback to internal English chat corpus so training still runs.

### Repeated outputs like `name name name`

Model is undertrained or trained on non-chat-heavy text.  
Increase training steps on quality conversational data and keep repetition penalty > 1.0.

### CUDA OOM

Reduce:
- `dim`
- `experts`
- `seq_len`
- `batch_size`

Or increase `grad_accum_steps`.

### Slow/limited HF downloads

Set `HF_TOKEN` for authenticated Hugging Face requests and better rate limits.

## Production Notes

This repository is an experimental research-oriented training stack, but the codebase includes several production-minded behaviors:
- explicit profile-based hardware scaling
- resumable stateful streaming ingestion
- failure-tolerant checkpoint writes
- safe train/infer concurrency
- deterministic config surfaces through `omnigenesis.yaml`

For production deployment, add:
- service API layer (instead of CLI)
- observability (metrics, traces, alerts)
- dataset governance and eval gates
- security hardening and sandboxed execution policy

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

Contributions should preserve:
- thread safety
- checkpoint compatibility
- profile-driven configurability
- clear tests for behavior changes

## License

MIT License. See [LICENSE](LICENSE).
