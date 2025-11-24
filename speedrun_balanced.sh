#!/bin/bash

# Balanced speed/quality version for single GPU training (1-2 hours on A10)
# This script aims to produce a model with basic reasoning capability while
# maintaining fast training time on a single GPU with 24GB memory.
#
# Compared to speedrun_test.sh (20-30 min, unusable model):
#   - 2.5x larger model (d10 vs d4)
#   - 125x more training iterations
#   - 5x better tokenizer
#   - Keeps evaluations enabled for quality monitoring
#
# Compared to speedrun.sh (4 hours, 8xH100, production model):
#   - 4x smaller model (d10 vs d20)
#   - Single GPU instead of 8
#   - ~136x fewer training tokens
#   - Good for testing, demos, and basic reasoning tasks

# 1) Example launch (simplest):
# bash speedrun_balanced.sh
# 2) Example launch in a screen session:
# screen -L -Logfile speedrun_balanced.log -S speedrun_balanced bash speedrun_balanced.sh
# 3) Example launch with wandb logging:
# WANDB_RUN=speedrun_balanced screen -L -Logfile speedrun_balanced.log -S speedrun_balanced bash speedrun_balanced.sh

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
#    `WANDB_RUN=speedrun_balanced bash speedrun_balanced.sh`
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

# Download initial dataset for tokenizer (8 shards = ~2B chars)
# Each shard is ~250M chars, so 8 shards covers initial tokenizer training
python -m nanochat.dataset -n 8
# Download more shards in background for pretraining (40 shards = ~10B chars)
# This is enough for our smaller model training
python -m nanochat.dataset -n 40 &
DATASET_DOWNLOAD_PID=$!

# Train the tokenizer with 500M chars (5x more than test, 1/4 of production)
# This gives much better vocabulary coverage than the test run
python -m scripts.tok_train --max_chars=500000000

# Evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Number of processes/GPUs to use
# Set to 1 for single GPU training
NPROC_PER_NODE=1

# Pretrain a balanced d10 model (depth=10, ~140M parameters)
# d10 model architecture:
#   - model_dim = 10 * 64 = 640
#   - num_heads = (640 + 127) // 128 = 6
#   - Parameters: ~140M (vs 37M test / 561M production)
#
# Training configuration optimized for 1xA10 (24GB):
#   - max_seq_len=1024: Better context than test (512), fits in memory
#   - device_batch_size=8: Fits comfortably in 24GB with gradient accumulation
#   - total_batch_size=32768: Reasonable batch size (62x larger than test)
#   - num_iterations=2500: 125x more training than test
#   - Training tokens: ~82M (4,100x more than test, 1/136 of production)
#
# With gradient accumulation, effective batch size per step is maintained
# while keeping memory usage under control.
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
  --depth=10 \
  --max_seq_len=1024 \
  --num_iterations=2500 \
  --device_batch_size=8 \
  --total_batch_size=32768 \
  --eval_every=250 \
  --eval_tokens=655360 \
  --core_metric_every=1000 \
  --sample_every=500 \
  --run=$WANDB_RUN

# Evaluate the model on train/val data
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss -- \
  --device_batch_size=8 \
  --split_tokens=65536

# Evaluate the model on CORE tasks (reduced problem count for speed)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --max-per-task=100

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# Download identity conversations dataset
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Run balanced midtraining (5x more iterations than test)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- \
  --num_iterations=250 \
  --max_seq_len=1024 \
  --device_batch_size=4 \
  --total_batch_size=8192 \
  --eval_every=100 \
  --run=$WANDB_RUN

# Evaluate mid model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- \
  -i mid \
  --max-problems=50 \
  --max-new-tokens=128

# -----------------------------------------------------------------------------
# Supervised Finetuning

# Train SFT with balanced iterations (4x more than test)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- \
  --device_batch_size=2 \
  --target_examples_per_step=16 \
  --num_iterations=200 \
  --eval_every=100 \
  --eval_steps=20 \
  --eval_metrics_every=200 \
  --run=$WANDB_RUN

# Evaluate SFT model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- \
  -i sft \
  --max-problems=50 \
  --max-new-tokens=128

# Chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report
python -m nanochat.report generate

echo "============================================"
echo "Balanced training run completed!"
echo ""
echo "Configuration summary:"
echo "  - 1 GPU (A10 or similar with 24GB)"
echo "  - depth=10 (~140M parameter model)"
echo "  - 2,500 base training iterations (~82M tokens)"
echo "  - 250 midtraining iterations"
echo "  - 200 SFT iterations"
echo "  - Full evaluations enabled"
echo ""
echo "Expected runtime: 1.5-2 hours"
echo "Expected quality: Basic reasoning, coherent responses"
echo ""
echo "To chat with your model:"
echo "  python -m scripts.chat_web"
echo "============================================"
