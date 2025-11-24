# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanochat is a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable codebase. It includes tokenization, pretraining, finetuning, evaluation, inference, and web serving. The goal is to be maximally forkable and readable, not an exhaustively configurable framework.

## Key Commands

### Environment Setup
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies (GPU version)
uv sync --extra gpu

# Install dependencies (CPU/MPS version for local dev)
uv sync --extra cpu

# Build the Rust tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Training Pipelines

**Production Run ($100 tier, ~4 hours on 8XH100)**
```bash
bash speedrun.sh
# Or in screen session:
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

**Fast Test Run (~20-30 minutes, for testing pipeline)**
```bash
bash speedrun_test.sh
```

**CPU/MPS Local Development**
```bash
bash dev/runcpu.sh
```

### Training Individual Stages

**Tokenizer**
```bash
# Train tokenizer on ~2B chars with vocab size 2^16
python -m scripts.tok_train --max_chars=2000000000

# Evaluate tokenizer compression
python -m scripts.tok_eval
```

**Base Model Pretraining**
```bash
# Single GPU
python -m scripts.base_train

# Multi-GPU (distributed)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train

# Custom depth (d26 example)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16

# Evaluate base model
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
```

**Midtraining** (conversation format, tool use, multiple choice)
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid
```

**Supervised Fine-Tuning (SFT)**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

**Reinforcement Learning** (optional, GSM8K only)
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K
```

### Inference

**CLI Chat**
```bash
# Interactive
python -m scripts.chat_cli

# Single prompt
python -m scripts.chat_cli -p "Why is the sky blue?"
```

**Web UI**
```bash
python -m scripts.chat_web
# Access at http://<IP>:8000/
```

### Testing
```bash
# Run specific test
python -m pytest tests/test_rustbpe.py -v -s

# Run all tests
python -m pytest tests/ -v -s
```

### Data Management
```bash
# Download N shards of pretraining data (~250M chars per shard)
python -m nanochat.dataset -n 240
```

## Architecture Overview

### Training Pipeline Stages

1. **Tokenizer Training** (`rustbpe/`, `nanochat/tokenizer.py`)
   - Custom Rust BPE tokenizer (efficient training, exports for tiktoken inference)
   - Trained on FineWeb pretraining data
   - Vocabulary size: 2^16 = 65,536 tokens
   - Target compression: ~4.8 chars/token

2. **Base Model Pretraining** (`scripts/base_train.py`)
   - Standard GPT pretraining on raw text
   - Follows Chinchilla scaling: 20x tokens per parameter
   - Example: d20 (561M params) needs ~11B tokens (240 data shards)

3. **Midtraining** (`scripts/mid_train.py`)
   - Teaches conversation format (special tokens)
   - Adds tool use capability
   - Trains on multiple choice format
   - Includes synthetic identity data to impart personality

4. **Supervised Fine-Tuning** (`scripts/chat_sft.py`)
   - Domain adaptation on conversational datasets
   - Each sequence is a complete conversation

5. **Reinforcement Learning** (`scripts/chat_rl.py`, optional)
   - Currently only supports GSM8K math reasoning
   - Uses policy gradient methods

### Model Architecture (`nanochat/gpt.py`)

The GPT implementation has these key features:
- **No positional embeddings**: Uses RoPE (Rotary Position Embeddings) instead
- **QK normalization**: Normalizes queries and keys in attention
- **Untied weights**: Separate token embedding and lm_head
- **ReLU² activation**: In MLP layers
- **Norm after embedding**: Layer norm after token embedding
- **No learnable params in rmsnorm**: Purely functional
- **No bias terms**: In linear layers
- **Group-Query Attention (GQA)**: For efficient inference with KV caching

Model sizing follows: `model_dim = depth * 64`, `num_heads = (model_dim + 127) // 128`

### Configuration System (`nanochat/configurator.py`)

Uses a unique "Poor Man's Configurator" approach instead of argparse:
- Settings are defined as global variables at the top of each script
- Override via config files: `python script.py path/to/config.py`
- Override via CLI args: `python script.py --depth=26 --device_batch_size=16`
- The configurator modifies `globals()` directly

### Distributed Training

All training scripts support both single-GPU and multi-GPU:
- **Single GPU**: Run script directly with `python -m scripts.script_name`
- **Multi-GPU**: Use `torchrun --standalone --nproc_per_node=N -m scripts.script_name`
- Automatic gradient accumulation if batch size doesn't fit in memory
- Reduce `--device_batch_size` if OOM (script compensates with more accumulation steps)

### Data Pipeline (`nanochat/dataloader.py`, `nanochat/dataset.py`)

- Pretraining data: FineWeb, packaged as ~250MB parquet shards
- Streaming dataloader with on-the-fly tokenization
- Distributed across GPUs (each GPU reads different row groups)
- Stateful: supports approximate training resumption

### Evaluation Tasks (`tasks/`)

Task system with two evaluation types:
- **categorical**: Multiple choice (ARC, MMLU)
- **generative**: Open-ended generation (GSM8K, HumanEval)

Tasks include:
- **CORE**: Base model benchmark (DCLM paper)
- **ARC-Challenge, ARC-Easy**: Science questions
- **MMLU**: Broad multiple choice knowledge
- **GSM8K**: Grade school math problems
- **HumanEval**: Python coding tasks
- **SmolTalk**: Conversational dataset (HuggingFace)
- **SpellingBee**: Custom task for spelling/counting

Tasks can be mixed for training via `TaskMixture` class.

### Optimizers

Two optimizers used:
- **Muon** (`nanochat/muon.py`): For weight matrices (higher LR ~0.02)
- **AdamW** (`nanochat/adamw.py`): For embeddings/unembeddings (lower LR ~0.004-0.2)

Both have distributed versions (`DistMuon`, `DistAdamW`) for DDP training.

### Inference Engine (`nanochat/engine.py`)

Efficient inference with:
- KV cache for autoregressive generation
- Supports tool execution (`nanochat/execution.py`)
- Temperature and sampling controls

### Checkpointing (`nanochat/checkpoint_manager.py`)

- Saves model state, optimizer state, and training metadata
- Stored in `~/.cache/nanochat/` by default (override with `NANOCHAT_BASE_DIR`)
- Format: `{base_dir}/{model_tag}/checkpoint_{step}.pt`

### Reporting (`nanochat/report.py`)

Generates `report.md` with:
- System info and timestamps
- All evaluation metrics across training stages
- Samples from the model
- Final summary table

## Memory Management

If you encounter OOM errors:
- Reduce `--device_batch_size` (e.g., 32 → 16 → 8 → 4)
- Reduce `--max_seq_len` if training
- For larger models (d26+), halve batch size: `--device_batch_size=16`
- Script automatically compensates with gradient accumulation

## Device Support

- **CUDA**: Primary target, full performance
- **MPS** (Apple Silicon): Supported but limited performance
- **CPU**: Supported but very slow, only for testing

The codebase auto-detects device type. Override with `--device_type=cuda|mps|cpu`

## Weights & Biases Integration

All training scripts support wandb logging:
```bash
# Login once
wandb login

# Run with logging
WANDB_RUN=my_experiment_name bash speedrun.sh
```

Use `WANDB_RUN=dummy` (default) to disable wandb logging.

## Customization

To customize nanochat personality:
1. Generate synthetic identity conversations (see `dev/gen_synthetic_data.py`)
2. Mix into midtraining and SFT stages via `TaskMixture`
3. See GitHub Discussions for detailed guides

## Code Style

- Minimal dependencies, vanilla PyTorch
- No configuration objects or model factories
- Single cohesive codebase, not a framework
- Prefer readability over exhaustive configurability
- No historical baggage or if-then-else monsters

## Important File Paths

- Training checkpoints: `~/.cache/nanochat/{model_tag}/checkpoint_{step}.pt`
- Tokenizer artifacts: `~/.cache/nanochat/tok{vocab_size}/`
- Pretraining data: `~/.cache/nanochat/fw_*.parquet`
- Generated report: `~/.cache/nanochat/report.md` (copied to repo root)
