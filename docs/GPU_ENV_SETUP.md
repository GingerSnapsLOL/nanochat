# GPU Environment Setup: CUDA_VISIBLE_DEVICES Guide

## Quick Answer

**YES** - If you set `CUDA_VISIBLE_DEVICES=""` for CPU training, you need to **remove/unset it** for GPU training.

---

## Understanding CUDA_VISIBLE_DEVICES

### What it does:
- `CUDA_VISIBLE_DEVICES=""` → Hides all GPUs (forces CPU)
- `CUDA_VISIBLE_DEVICES="0"` → Shows only GPU 0
- `CUDA_VISIBLE_DEVICES="0,1"` → Shows GPUs 0 and 1
- **Not set** → Shows all available GPUs (default)

---

## How to Check Current Setting

```bash
# In WSL, check if it's set:
echo $CUDA_VISIBLE_DEVICES

# If it shows nothing (empty), it's not set (good for GPU)
# If it shows "", it's set to empty (bad for GPU - hides GPUs)
```

---

## For GPU Training: Remove/Unset It

### Option 1: Unset for Current Session
```bash
# Unset the variable
unset CUDA_VISIBLE_DEVICES

# Verify it's gone
echo $CUDA_VISIBLE_DEVICES
# Should show nothing (not set)
```

### Option 2: Remove from ~/.bashrc (if you added it there)
```bash
# Check if it's in your bashrc
grep CUDA_VISIBLE_DEVICES ~/.bashrc

# If found, remove that line
nano ~/.bashrc
# Delete or comment out the line: export CUDA_VISIBLE_DEVICES=""
# Save and exit (Ctrl+X, Y, Enter)

# Reload bashrc
source ~/.bashrc
```

### Option 3: Override for Specific Command
```bash
# Run command without the variable
env -u CUDA_VISIBLE_DEVICES uv run -m scripts.base_train --depth=16
```

---

## Complete GPU Setup Checklist

### 1. Remove CPU-only setting
```bash
# Unset for current session
unset CUDA_VISIBLE_DEVICES

# Remove from ~/.bashrc if you added it
sed -i '/CUDA_VISIBLE_DEVICES/d' ~/.bashrc
source ~/.bashrc
```

### 2. Verify GPU is visible
```bash
# Should show your GPU
nvidia-smi

# Should return True
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### 3. Use GPU PyTorch
```bash
# Make sure you have GPU version
uv sync --extra gpu  # NOT --extra cpu
```

### 4. Train with GPU
```bash
# Now GPU will be used automatically
uv run -m scripts.base_train --depth=16
```

---

## Quick Reference

| Scenario | CUDA_VISIBLE_DEVICES | Command |
|----------|---------------------|---------|
| **CPU Training** | `export CUDA_VISIBLE_DEVICES=""` | `uv sync --extra cpu` |
| **GPU Training** | `unset CUDA_VISIBLE_DEVICES` | `uv sync --extra gpu` |
| **Single GPU** | Not set (default) or `export CUDA_VISIBLE_DEVICES="0"` | `uv sync --extra gpu` |
| **Multiple GPUs** | Not set (default) or `export CUDA_VISIBLE_DEVICES="0,1"` | `uv sync --extra gpu` |

---

## Testing GPU Access

```bash
# Test 1: Check nvidia-smi
nvidia-smi
# Should show your GPU(s)

# Test 2: Check PyTorch CUDA
uv run python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

**Expected output for GPU:**
```
CUDA available: True
GPU count: 1
GPU name: NVIDIA GeForce RTX ...
```

**If you see `CUDA available: False`:**
1. Check `CUDA_VISIBLE_DEVICES` is not set to ""
2. Verify `nvidia-smi` works
3. Make sure you used `uv sync --extra gpu`

---

## Common Issues

### Issue: "CUDA available: False" even with GPU

**Solution:**
```bash
# 1. Unset the variable
unset CUDA_VISIBLE_DEVICES

# 2. Check nvidia-smi works
nvidia-smi

# 3. Reinstall GPU PyTorch
uv sync --extra gpu

# 4. Test again
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: "CUDA out of memory"

**Solution:**
- Reduce batch size: `--device_batch_size=8` (or 4, 2)
- Reduce model size: `--depth=12` (instead of 16 or 20)
- Reduce sequence length: `--max_seq_len=1024`

### Issue: Want to switch between CPU and GPU

**Solution:**
```bash
# For CPU training
export CUDA_VISIBLE_DEVICES=""
uv sync --extra cpu
uv run -m scripts.base_train --depth=4

# For GPU training
unset CUDA_VISIBLE_DEVICES
uv sync --extra gpu
uv run -m scripts.base_train --depth=16
```

---

## Summary

✅ **For GPU training**: `unset CUDA_VISIBLE_DEVICES` (or don't set it)  
✅ **For CPU training**: `export CUDA_VISIBLE_DEVICES=""`  
✅ **Always use**: `uv sync --extra gpu` for GPU, `--extra cpu` for CPU  
✅ **Test first**: `uv run python -c "import torch; print(torch.cuda.is_available())"`

The key is: **unset the variable** before GPU training!

