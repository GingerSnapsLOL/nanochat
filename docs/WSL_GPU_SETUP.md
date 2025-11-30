# GPU Training in WSL2 - Complete Guide

## ✅ Good News: You CAN Use GPU in WSL2!

WSL2 **fully supports** NVIDIA GPU training! You don't need "real Ubuntu" - WSL2 works perfectly for GPU training. The confusion might be that you need to set up CUDA properly.

## Why Your Model is "Not OK" (Expected!)

The `dev/runcpu.sh` script trains a **tiny model on CPU** for learning purposes. The model quality will be poor because:
- It's only 4 layers (depth=4) - very small
- Training on CPU is extremely slow
- Only 50 training iterations (not enough)
- This is just for **learning the code paths**, not producing a good model

For a real model, you need:
- GPU acceleration
- More layers (depth=20+)
- More training iterations
- More data

---

## Setting Up GPU Training in WSL2

### Prerequisites

1. **NVIDIA GPU** (required)
2. **NVIDIA Drivers on Windows** (must be installed first!)
3. **WSL2** (you already have this)

### Step-by-Step Setup

#### Step 1: Install NVIDIA Drivers on Windows

1. Download latest NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx
2. Install them on Windows (not in WSL)
3. Restart your computer

#### Step 2: Verify GPU is Visible in WSL

Open WSL and run:
```bash
# This should show your GPU
nvidia-smi
```

If `nvidia-smi` works, you're ready! If not, you need to install CUDA toolkit in WSL.

#### Step 3: Install CUDA Toolkit in WSL (if needed)

```bash
# In WSL Ubuntu:
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

# Verify
nvidia-smi
nvcc --version
```

#### Step 4: Reinstall PyTorch with GPU Support

```bash
# In your nanochat directory
cd /mnt/c/Users/6204/Desktop/nanochat

# Remove CPU-only environment
rm -rf .venv

# Create new environment with GPU support
uv venv
uv sync --extra gpu

# Test GPU
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

You should see:
```
CUDA available: True
GPU: NVIDIA GeForce RTX ... (your GPU name)
```

---

## Training Your Own Model on One GPU

Once GPU is set up, you can train a real model! Here's how:

### Option 1: Use the Speedrun Script (Modified for 1 GPU)

Create `train_1gpu.sh`:

```bash
#!/bin/bash

# Single GPU training script for WSL2
# Run as: bash train_1gpu.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Setup
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu  # Note: GPU, not CPU!

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Install Rust if needed
command -v rustc &> /dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env" 2>/dev/null || true

# Build tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Reset report
uv run -m nanochat.report reset

# Download data (fewer shards for 1 GPU - adjust based on your GPU memory)
uv run -m nanochat.dataset -n 8
uv run -m nanochat.dataset -n 240 &

# Train tokenizer
uv run -m scripts.tok_train --max_chars=2000000000
uv run -m scripts.tok_eval

# Wait for data download
wait

# Train base model (single GPU, smaller batch size)
# Adjust these based on your GPU memory:
# - depth: Model size (12-20 for 1 GPU)
# - device_batch_size: Reduce if OOM (try 16, 8, 4, or 2)
# - max_seq_len: Context length (2048 is good)

uv run -m scripts.base_train \
    --depth=16 \
    --max_seq_len=2048 \
    --device_batch_size=16 \
    --total_batch_size=524288 \
    --run=$WANDB_RUN

# Evaluate
uv run -m scripts.base_loss
uv run -m scripts.base_eval

# Download identity conversations
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Midtraining
uv run -m scripts.mid_train -- --run=$WANDB_RUN
uv run -m scripts.chat_eval -- -i mid

# SFT
uv run -m scripts.chat_sft -- --run=$WANDB_RUN
uv run -m scripts.chat_eval -- -i sft

# Generate report
uv run -m nanochat.report generate

echo "Training complete! Chat with your model:"
echo "uv run -m scripts.chat_web"
```

### Option 2: Manual Training Commands

```bash
# Single GPU training (no torchrun needed)
uv run -m scripts.base_train \
    --depth=16 \
    --device_batch_size=16 \
    --max_seq_len=2048

# For evaluation
uv run -m scripts.base_eval
```

---

## Customizing Your Model

### Adjust Model Size (depth)

```bash
# Smaller model (faster, less memory)
--depth=12

# Medium model (good balance)
--depth=16

# Larger model (better quality, needs more memory)
--depth=20
```

### Adjust Batch Size (if you get OOM errors)

```bash
# If you get "Out of Memory" errors, reduce batch size:
--device_batch_size=8   # Try 8
--device_batch_size=4   # Or 4
--device_batch_size=2   # Or even 2
```

The script automatically handles gradient accumulation to maintain the total batch size.

### Custom Optimizer

You can modify the optimizer in `nanochat/gpt.py`:
- Look for `setup_optimizers()` method
- Adjust learning rates: `embedding_lr`, `unembedding_lr`, `matrix_lr`
- Or modify `nanochat/adamw.py` and `nanochat/muon.py`

### Custom Tokenizer

The tokenizer is in `rustbpe/` directory:
- Modify `rustbpe/src/lib.rs` for tokenizer logic
- Rebuild with: `uv run maturin develop --release --manifest-path rustbpe/Cargo.toml`

---

## Troubleshooting

### "CUDA out of memory" Error

Reduce batch size:
```bash
--device_batch_size=8  # or 4, 2, 1
```

Or reduce model size:
```bash
--depth=12  # instead of 16 or 20
```

### "nvidia-smi: command not found"

1. Install NVIDIA drivers on Windows
2. Restart computer
3. Try `nvidia-smi` in WSL again

### "CUDA available: False"

1. Make sure you installed CUDA toolkit in WSL (Step 3 above)
2. Reinstall PyTorch: `uv sync --extra gpu`
3. Check: `uv run python -c "import torch; print(torch.cuda.is_available())"`

### Model Quality is Poor

This is normal if:
- Training for too few iterations
- Model too small (depth < 12)
- Not enough data
- Training on CPU (very slow)

For better quality:
- Use GPU
- Train longer (more iterations)
- Use depth=16 or 20
- Ensure enough data shards downloaded

---

## Summary

✅ **WSL2 supports GPU training** - you don't need real Ubuntu  
✅ **CPU training is just for learning** - model quality will be poor  
✅ **GPU training produces real models** - follow the setup steps above  
✅ **Single GPU works fine** - adjust batch size and model depth as needed  

The key is setting up CUDA in WSL2, then using `--extra gpu` instead of `--extra cpu`!

