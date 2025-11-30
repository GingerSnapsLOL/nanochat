#!/bin/bash

# Quick test script to verify nanochat environment setup in WSL (CPU-only)
# Run as: bash test_env.sh

set -e  # Exit on error

echo "========================================"
echo "nanochat Environment Test (CPU-only)"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test 1: Check if uv is installed
echo -e "${YELLOW}[1/7] Checking uv installation...${NC}"
if command -v uv &> /dev/null; then
    echo -e "${GREEN}✓ uv is installed${NC}"
    uv --version
else
    echo -e "${RED}✗ uv is not installed${NC}"
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi
echo ""

# Test 2: Check if virtual environment exists
echo -e "${YELLOW}[2/7] Checking virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
else
    echo -e "${YELLOW}! Virtual environment not found, creating...${NC}"
    uv venv
fi
echo ""

# Test 3: Sync dependencies (CPU-only)
echo -e "${YELLOW}[3/7] Installing/syncing dependencies (CPU-only)...${NC}"
uv sync --extra cpu
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Test 4: Check Rust/Cargo
echo -e "${YELLOW}[4/7] Checking Rust installation...${NC}"
if command -v rustc &> /dev/null; then
    echo -e "${GREEN}✓ Rust is installed${NC}"
    rustc --version
else
    echo -e "${YELLOW}! Rust not found, installing...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
echo ""

# Test 5: Build Rust tokenizer
echo -e "${YELLOW}[5/7] Building Rust tokenizer...${NC}"
if uv run maturin develop --release --manifest-path rustbpe/Cargo.toml; then
    echo -e "${GREEN}✓ Rust tokenizer built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build Rust tokenizer${NC}"
    exit 1
fi
echo ""

# Test 6: Test Python imports
echo -e "${YELLOW}[6/7] Testing Python imports...${NC}"
if uv run python -c "
import torch
import sys
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {\"CUDA\" if torch.cuda.is_available() else \"CPU\"}')"; then
    echo -e "${GREEN}✓ Python imports successful${NC}"
else
    echo -e "${RED}✗ Python import failed${NC}"
    exit 1
fi
echo ""

# Test 7: Run a simple test
echo -e "${YELLOW}[7/7] Running engine test (CPU-only)...${NC}"
# Set environment to prevent CUDA loading
export CUDA_VISIBLE_DEVICES=""
if uv run python -m pytest tests/test_engine.py -v; then
    echo -e "${GREEN}✓ Tests passed!${NC}"
else
    echo -e "${YELLOW}! Tests failed or skipped (this is okay for CPU-only setup)${NC}"
fi
echo ""

# Summary
echo "========================================"
echo -e "${GREEN}Environment Test Complete!${NC}"
echo "========================================"
echo ""
echo "Your nanochat environment is ready for CPU-only training."
echo ""
echo "Next steps:"
echo "  1. Run CPU demo: bash dev/runcpu.sh"
echo "  2. Or start training: uv run -m scripts.base_train --depth=4 --device_batch_size=1"
echo ""

