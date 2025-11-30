#!/bin/bash

# Single GPU training script for WSL2
# This trains a real model (not just for learning)
# Run as: bash train_1gpu.sh

set -e  # Exit on error

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}nanochat Single GPU Training${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Setup uv
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}[1/9] Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env" 2>/dev/null || true
else
    echo -e "${GREEN}[1/9]✓ uv already installed${NC}"
fi

# Setup virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}[2/9] Creating virtual environment...${NC}"
    uv venv
else
    echo -e "${GREEN}[2/9]✓ Virtual environment already exists${NC}"
fi

# Sync dependencies (uv sync is idempotent - it checks what's needed)
echo -e "${YELLOW}[2/9] Syncing dependencies (uv will skip if already installed)...${NC}"
uv sync --extra gpu
echo -e "${GREEN}[2/9]✓ Dependencies synced${NC}"

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Install Rust if needed
if ! command -v rustc &> /dev/null; then
    echo -e "${YELLOW}[3/9] Installing Rust...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env" 2>/dev/null || true
    echo -e "${GREEN}[3/9]✓ Rust installed${NC}"
else
    echo -e "${GREEN}[3/9]✓ Rust already installed${NC}"
fi

# Build tokenizer (check if rustbpe module is importable)
echo -e "${YELLOW}[4/9] Checking Rust tokenizer...${NC}"
if uv run python -c "import rustbpe" 2>/dev/null; then
    echo -e "${GREEN}[4/9]✓ Rust tokenizer already built${NC}"
else
    echo -e "${YELLOW}[4/9] Building Rust tokenizer...${NC}"
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
    echo -e "${GREEN}[4/9]✓ Rust tokenizer built${NC}"
fi

# Reset report (always reset for fresh run)
echo -e "${YELLOW}[5/9] Resetting report...${NC}"
uv run -m nanochat.report reset
echo -e "${GREEN}[5/9]✓ Report reset${NC}"

# Download data (dataset.py already skips existing files, but we check count)
echo -e "${YELLOW}[6/9] Checking dataset...${NC}"
DATA_DIR="$NANOCHAT_BASE_DIR/base_data"
mkdir -p "$DATA_DIR"
EXISTING_SHARDS=$(find "$DATA_DIR" -name "shard_*.parquet" 2>/dev/null | wc -l)
NEEDED_SHARDS=240

if [ "$EXISTING_SHARDS" -ge "$NEEDED_SHARDS" ]; then
    echo -e "${GREEN}[6/9]✓ Dataset already downloaded ($EXISTING_SHARDS/$NEEDED_SHARDS shards)${NC}"
    DATASET_DOWNLOAD_PID=""
else
    echo -e "${YELLOW}[6/9] Downloading dataset ($EXISTING_SHARDS/$NEEDED_SHARDS shards exist, dataset.py will skip existing)...${NC}"
    # Download initial 8 shards synchronously
    if [ "$EXISTING_SHARDS" -lt 8 ]; then
        uv run -m nanochat.dataset -n 8
    fi
    # Download remaining shards in background
    uv run -m nanochat.dataset -n $NEEDED_SHARDS &
    DATASET_DOWNLOAD_PID=$!
    echo -e "${GREEN}[6/9]✓ Dataset download started in background${NC}"
fi

# Train tokenizer (check if tokenizer.pkl exists)
echo -e "${YELLOW}[7/9] Checking tokenizer...${NC}"
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
TOKENIZER_PATH="$TOKENIZER_DIR/tokenizer.pkl"
if [ -f "$TOKENIZER_PATH" ]; then
    echo -e "${GREEN}[7/9]✓ Tokenizer already trained${NC}"
    echo -e "${YELLOW}[7/9] Evaluating tokenizer...${NC}"
    uv run -m scripts.tok_eval
else
    echo -e "${YELLOW}[7/9] Training tokenizer...${NC}"
    uv run -m scripts.tok_train --max_chars=2000000000
    uv run -m scripts.tok_eval
    echo -e "${GREEN}[7/9]✓ Tokenizer trained${NC}"
fi

# Wait for data download if it was started
if [ -n "$DATASET_DOWNLOAD_PID" ]; then
    echo -e "${YELLOW}[8/9] Waiting for dataset download to complete...${NC}"
    wait $DATASET_DOWNLOAD_PID
    echo -e "${GREEN}[8/9]✓ Dataset download complete${NC}"
else
    echo -e "${GREEN}[8/9]✓ Dataset ready${NC}"
fi

echo -e "${GREEN}[9/9]✓ Setup complete!${NC}"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Training${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Train base model (single GPU)
# Adjust these parameters based on your GPU memory:
# - depth: 12 (small), 16 (medium), 20 (large)
# - device_batch_size: Reduce if OOM (try 16, 8, 4, or 2)
# - max_seq_len: 2048 is good for most GPUs

echo "Starting base training on single GPU..."
uv run -m scripts.base_train \
    --depth=16 \
    --max_seq_len=2048 \
    --device_batch_size=16 \
    --total_batch_size=524288 \
    --run=$WANDB_RUN

# Evaluate base model
uv run -m scripts.base_loss
uv run -m scripts.base_eval

# Download identity conversations (check if exists)
IDENTITY_PATH="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ -f "$IDENTITY_PATH" ]; then
    echo -e "${GREEN}✓ Identity conversations already downloaded${NC}"
else
    echo -e "${YELLOW}Downloading identity conversations...${NC}"
    curl -L -o "$IDENTITY_PATH" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    echo -e "${GREEN}✓ Identity conversations downloaded${NC}"
fi

# Midtraining
echo ""
echo -e "${BLUE}Starting midtraining...${NC}"
uv run -m scripts.mid_train -- --run=$WANDB_RUN
uv run -m scripts.chat_eval -- -i mid

# SFT
echo ""
echo -e "${BLUE}Starting SFT...${NC}"
uv run -m scripts.chat_sft -- --run=$WANDB_RUN
uv run -m scripts.chat_eval -- -i sft

# Generate report
uv run -m nanochat.report generate

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Chat with your model:"
echo "  uv run -m scripts.chat_web"
echo ""

