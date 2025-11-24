#!/bin/bash

# Ultra-fast test version of speedrun.sh
# This script is designed to complete in ~20-30 minutes for end-to-end testing.
# It produces a tiny but functional LLM to verify the entire pipeline works.

# 1) Example launch (simplest):
# bash speedrun_test.sh
# 2) Example launch in a screen session:
# screen -L -Logfile speedrun_test.log -S speedrun_test bash speedrun_test.sh
# 3) Example launch with wandb logging:
# WANDB_RUN=speedrun_test screen -L -Logfile speedrun_test.log -S speedrun_test bash speedrun_test.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=speedrun_test bash speedrun_test.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download minimal dataset for testing (4 shards = ~400MB instead of 24GB)
# Each shard is ~250M chars, so 4 shards = ~1B chars
python -m nanochat.dataset -n 4

# Train the tokenizer with reduced data (100M chars instead of 2B)
# This is sufficient for testing the pipeline
python -m scripts.tok_train --max_chars=100000000

# Evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Number of processes/GPUs to use
# Set to 1 for single GPU, 8 for 8XH100, etc.
NPROC_PER_NODE=1

# Pretrain a minimal d4 model (depth=4) with very few iterations
# This uses:
# - depth=4 (tiny 4-layer model instead of 20-layer)
# - max_seq_len=512 (shorter sequences)
# - num_iterations=20 (explicit iteration count)
# - device_batch_size=1, total_batch_size=1024 (smaller batches)
# - core_metric_every=-1, sample_every=-1 (skip slow evaluations)
# - eval_every=20, eval_tokens=4096 (minimal validation)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
  --depth=4 \
  --max_seq_len=512 \
  --num_iterations=20 \
  --device_batch_size=1 \
  --total_batch_size=1024 \
  --eval_every=20 \
  --eval_tokens=4096 \
  --core_metric_every=-1 \
  --sample_every=-1 \
  --run=$WANDB_RUN

# Evaluate the model on minimal train/val data
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss -- \
  --device_batch_size=1 \
  --split_tokens=4096

# Skip base_eval (CORE tasks) to save time - uncomment if you want to run it
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --max-per-task=16

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# Download identity conversations dataset
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Run minimal midtraining
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- \
  --num_iterations=50 \
  --max_seq_len=1024 \
  --device_batch_size=1 \
  --total_batch_size=1024 \
  --eval_every=-1 \
  --run=$WANDB_RUN

# Evaluate mid model with minimal problems
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- \
  -i mid \
  --max-problems=20 \
  --max-new-tokens=64

# -----------------------------------------------------------------------------
# Supervised Finetuning

# Train SFT with minimal iterations
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- \
  --device_batch_size=1 \
  --target_examples_per_step=4 \
  --num_iterations=50 \
  --eval_every=50 \
  --eval_steps=4 \
  --eval_metrics_every=-1 \
  --run=$WANDB_RUN

# Evaluate SFT model with minimal problems
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- \
  -i sft \
  --max-problems=20 \
  --max-new-tokens=64

# Chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report
python -m nanochat.report generate

echo "============================================"
echo "Ultra-fast test run completed!"
echo "This was a minimal test run with:"
echo "  - 1 GPU (set NPROC_PER_NODE=8 for 8 GPUs)"
echo "  - depth=4 (tiny model)"
echo "  - 20 base training iterations"
echo "  - 50 midtraining iterations"
echo "  - 50 SFT iterations"
echo "  - Minimal evaluations"
echo ""
echo "To chat with your model:"
echo "  python -m scripts.chat_web"
echo "============================================"
