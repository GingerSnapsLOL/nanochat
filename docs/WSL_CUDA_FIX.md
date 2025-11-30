# Fixing CUDA Library Errors in WSL

## Problem
You're getting errors like:
```
ValueError: libcublas.so.*[0-9] not found
OSError: libcudart.so.12: cannot open shared object file
```

This happens because PyTorch was installed with GPU support (`--extra gpu`), but CUDA libraries aren't installed in WSL.

## Quick Fixes

### Option 1: Set Environment Variable (Easiest - For Testing)

Before running tests or scripts, set this environment variable:

```bash
# In WSL, before running anything:
export CUDA_VISIBLE_DEVICES=""
```

Or add it to your `~/.bashrc` to make it permanent:
```bash
echo 'export CUDA_VISIBLE_DEVICES=""' >> ~/.bashrc
source ~/.bashrc
```

Then run your tests:
```bash
python -m pytest tests/test_engine.py -v
```

### Option 2: Reinstall with CPU-Only PyTorch (For Learning)

If you don't have a GPU or just want to learn without GPU:

```bash
# Remove current environment
rm -rf .venv

# Recreate with CPU support
uv venv
source .venv/bin/activate
uv sync --extra cpu
```

Now tests should work:
```bash
python -m pytest tests/test_engine.py -v
```

### Option 3: Install CUDA in WSL (If you have NVIDIA GPU)

If you have an NVIDIA GPU and want to use it for training:

1. **Install NVIDIA drivers on Windows** (if not already):
   - Download from: https://www.nvidia.com/Download/index.aspx
   - Install the latest drivers

2. **Install CUDA Toolkit in WSL**:
   ```bash
   # In WSL Ubuntu:
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-4
   ```

3. **Verify installation**:
   ```bash
   nvidia-smi  # Should show your GPU
   ```

4. **Reinstall PyTorch** (if needed):
   ```bash
   uv sync --extra gpu
   ```

## Recommended Approach

**For learning/testing**: Use Option 2 (CPU-only) or Option 1 (environment variable)

**For actual training**: Use Option 3 (install CUDA) if you have a GPU

## What I Fixed

I've updated `tests/test_engine.py` to set `CUDA_VISIBLE_DEVICES=""` by default, which should prevent the CUDA loading error. However, the best long-term solution depends on your setup:

- **No GPU or just learning**: Use CPU-only PyTorch
- **Have GPU and want to train**: Install CUDA in WSL
