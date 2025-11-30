# Model Selection Guide: GPT vs Alcoholic

## Overview

The training script now supports two model architectures:
- **GPT**: The standard nanochat GPT model (default)
- **Alcoholic**: An alternative architecture with different features

## Quick Start

### Use GPT Model (Default)
```bash
uv run -m scripts.base_train --depth=20
# or explicitly:
uv run -m scripts.base_train --model_type=gpt --depth=20
```

### Use Alcoholic Model
```bash
uv run -m scripts.base_train --model_type=alcoholic --depth=20
```

## Model Differences

### GPT Model (Standard)
- **Activation**: ReLU² (relu squared)
- **Normalization**: Functional RMSNorm (no learnable params)
- **MLP**: Standard feedforward (4x expansion)
- **RoPE Base**: 10,000
- **Architecture**: Simpler, proven design

### Alcoholic Model
- **Activation**: SwiGLU (SiLU + Gated Linear Unit)
- **Normalization**: Learnable RMSNorm (with weight parameter)
- **MLP**: SwiGLU-based (configurable intermediate size)
- **RoPE Base**: 1,000,000 (configurable via `rope_theta`)
- **Features**:
  - Optional QK normalization
  - Input/output normalization layers
  - More configurable architecture

## Configuration

### GPT Model Config
```python
GPTConfig(
    sequence_len=2048,
    vocab_size=65536,
    n_layer=20,
    n_head=16,
    n_kv_head=16,
    n_embd=1280,
)
```

### Alcoholic Model Config
```python
AlcoholicNanoConfig(
    sequence_len=2048,
    vocab_size=65536,
    n_layer=20,
    n_head=16,
    n_kv_head=8,  # Typically half of n_head
    n_embd=1280,
    intermediate_size=5120,  # 4x n_embd
    rope_theta=1_000_000.0,
    qk_norm=True,
    norm_type="rmsnorm",
    mlp_type="swiglu",
    dropout=0.0,
)
```

## Training Parameters

Both models use the same training parameters:
- `--depth`: Number of layers
- `--max_seq_len`: Context length
- `--device_batch_size`: Batch size per device
- `--total_batch_size`: Effective batch size
- Learning rates, etc.

## When to Use Which?

### Use GPT (Default)
- ✅ Standard, proven architecture
- ✅ Simpler and faster
- ✅ Better documented
- ✅ Recommended for most use cases
- ✅ Lower memory usage

### Use Alcoholic
- ✅ Want to experiment with SwiGLU
- ✅ Need longer context (higher rope_theta)
- ✅ Want learnable normalization
- ✅ Research/experimentation
- ⚠️ More memory intensive
- ⚠️ Less tested

## Examples

### Small Model (Learning)
```bash
# GPT
uv run -m scripts.base_train --model_type=gpt --depth=4 --device_batch_size=1 --num_iterations=50

# Alcoholic
uv run -m scripts.base_train --model_type=alcoholic --depth=4 --device_batch_size=1 --num_iterations=50
```

### Medium Model (Single GPU)
```bash
# GPT
uv run -m scripts.base_train --model_type=gpt --depth=16 --device_batch_size=16

# Alcoholic
uv run -m scripts.base_train --model_type=alcoholic --depth=16 --device_batch_size=16
```

### Large Model (Multi-GPU)
```bash
# GPT
uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --model_type=gpt --depth=20

# Alcoholic
uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --model_type=alcoholic --depth=20
```

## Checkpoint Naming

Checkpoints are saved with model type in the name:
- GPT: `base_checkpoints/gpt_d20/`
- Alcoholic: `base_checkpoints/alcoholic_d20/`

This prevents accidentally loading the wrong model type.

## Compatibility

Both models are compatible with:
- ✅ Same training script
- ✅ Same data loader
- ✅ Same optimizer setup (AdamW + Muon)
- ✅ Same evaluation scripts
- ✅ Same inference engine

## Troubleshooting

### "CUDA out of memory" with Alcoholic
Alcoholic uses more memory. Try:
- Reduce `--device_batch_size` (e.g., 16 → 8)
- Reduce `--depth` (e.g., 20 → 16)
- Reduce `--max_seq_len` (e.g., 2048 → 1024)

### Model not loading correctly
Make sure you use the same `--model_type` when loading:
- If you trained with `--model_type=alcoholic`, use the same when loading
- Checkpoints include `model_type` in metadata

## Implementation Details

The training script automatically:
1. Detects `model_type` parameter
2. Creates appropriate config (GPTConfig or AlcoholicNanoConfig)
3. Initializes correct model class
4. Uses same training loop for both
5. Saves model type in checkpoint metadata

Both models implement the same interface:
- `forward(idx, targets, kv_cache, loss_reduction)`
- `init_weights()`
- `setup_optimizers(...)`
- `estimate_flops()`
- `generate(...)`

This ensures full compatibility with the training infrastructure.

