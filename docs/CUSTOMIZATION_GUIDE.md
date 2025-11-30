# Customization Guide: Creating Your Own Model Components

This guide shows you how to customize the model, optimizer, and tokenizer for your needs.

## 🎯 Quick Answer to Your Questions

### Can I use GPU in WSL2?
**YES!** WSL2 fully supports NVIDIA GPU training. You don't need "real Ubuntu". Just:
1. Install NVIDIA drivers on Windows
2. Install CUDA toolkit in WSL2
3. Use `uv sync --extra gpu` instead of `--extra cpu`

### Why is my model "not OK"?
The `dev/runcpu.sh` script trains a **tiny model on CPU** for learning:
- Only 4 layers (very small)
- Only 50 iterations (not enough)
- CPU training (extremely slow)
- This is just to learn the code, not produce a good model

For a real model, use GPU training with `train_1gpu.sh` (I just created this for you!)

---

## 🛠️ Customizing Components

### 1. Custom Model Architecture

**File**: `nanochat/gpt.py`

#### Change Model Depth (Number of Layers)
```python
# In scripts/base_train.py or your training script:
--depth=16  # 12=small, 16=medium, 20=large, 24+=very large
```

#### Change Model Dimensions
Edit `nanochat/gpt.py`:
```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048      # Context length
    vocab_size: int = 65536        # Vocabulary size
    n_layer: int = 16             # Number of transformer layers
    n_head: int = 16              # Number of attention heads
    n_kv_head: int = 4            # Key/value heads (for efficiency)
    n_embd: int = 1024            # Embedding dimension
```

#### Modify Attention Mechanism
In `nanochat/gpt.py`, find `CausalSelfAttention` class:
```python
class CausalSelfAttention(nn.Module):
    # Modify attention logic here
    # Change from Multi-Query to Multi-Head, etc.
```

#### Change Activation Function
Find the MLP block in `nanochat/gpt.py`:
```python
# Current: relu^2 activation
# Change to: GELU, Swish, etc.
self.act = lambda x: F.relu(x) ** 2  # Current
# self.act = F.gelu  # Alternative
```

---

### 2. Custom Optimizer

**Files**: `nanochat/adamw.py`, `nanochat/muon.py`, `nanochat/gpt.py`

#### Modify Learning Rates
In your training script or `nanochat/gpt.py`:
```python
# In setup_optimizers() method:
embedding_lr = 0.2      # Learning rate for embeddings
unembedding_lr = 0.004  # Learning rate for output head
matrix_lr = 0.02        # Learning rate for transformer layers
```

#### Create Custom Optimizer
Create new file `nanochat/custom_optimizer.py`:
```python
import torch
from torch.optim import Optimizer

class MyCustomOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, **kwargs):
        defaults = dict(lr=lr, **kwargs)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        # Your custom optimization logic
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # Custom update rule
                    p.data.add_(p.grad, alpha=-group['lr'])
```

Then modify `nanochat/gpt.py` to use it:
```python
def setup_optimizers(self, ...):
    # Replace AdamW/Muon with your optimizer
    custom_opt = MyCustomOptimizer(...)
    return custom_opt
```

#### Modify Existing Optimizers
- **AdamW**: Edit `nanochat/adamw.py`
- **Muon**: Edit `nanochat/muon.py`

---

### 3. Custom Tokenizer

**Files**: `rustbpe/src/lib.rs`, `nanochat/tokenizer.py`

#### Change Vocabulary Size
In `scripts/tok_train.py` or when training:
```bash
# Default is 65536 (2^16)
# Change in rustbpe/src/lib.rs or training script
```

#### Modify BPE Training
Edit `rustbpe/src/lib.rs`:
```rust
// Modify BPE algorithm
// Change merge rules, special tokens, etc.
```

Then rebuild:
```bash
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

#### Add Custom Special Tokens
In `nanochat/tokenizer.py`, find special token definitions:
```python
# Add your custom tokens
SPECIAL_TOKENS = {
    "<|user|>": ...,
    "<|assistant|>": ...,
    "<|your_token|>": ...,  # Add here
}
```

---

### 4. Custom Training Loop

**File**: `scripts/base_train.py`

#### Modify Training Steps
Edit the training loop in `scripts/base_train.py`:
```python
# Find the training loop (around line 180)
for step in range(num_iterations + 1):
    # Add custom logging
    # Add custom loss computation
    # Add custom evaluation
    # etc.
```

#### Add Custom Loss Function
Create `nanochat/custom_loss.py`:
```python
import torch
import torch.nn.functional as F

def custom_loss(logits, targets):
    # Your custom loss function
    base_loss = F.cross_entropy(logits, targets)
    # Add regularization, etc.
    return base_loss
```

Then use it in training:
```python
from nanochat.custom_loss import custom_loss
loss = custom_loss(logits, targets)
```

---

## 📝 Example: Complete Custom Model

Here's how to create a completely custom setup:

### Step 1: Create Custom Config
```python
# my_model_config.py
from nanochat.gpt import GPTConfig

class MyModelConfig(GPTConfig):
    n_layer = 12
    n_head = 12
    n_embd = 768
    sequence_len = 4096  # Longer context
```

### Step 2: Create Custom Training Script
```python
# scripts/my_train.py
from nanochat.gpt import GPT
from my_model_config import MyModelConfig

config = MyModelConfig()
model = GPT(config)
# ... rest of training code
```

### Step 3: Train
```bash
uv run -m scripts.my_train --your-custom-args
```

---

## 🎨 Practical Examples

### Example 1: Smaller Model for Faster Training
```bash
uv run -m scripts.base_train \
    --depth=12 \
    --device_batch_size=32 \
    --max_seq_len=1024
```

### Example 2: Larger Model for Better Quality
```bash
uv run -m scripts.base_train \
    --depth=20 \
    --device_batch_size=8 \
    --max_seq_len=2048
```

### Example 3: Custom Learning Rate Schedule
Edit `scripts/base_train.py`:
```python
def get_lr_multiplier(step):
    # Custom schedule
    if step < 1000:
        return step / 1000  # Warmup
    else:
        return 0.5 ** ((step - 1000) / 10000)  # Exponential decay
```

---

## 🔧 Tips for Customization

1. **Start Small**: Test changes with tiny models first
2. **One Change at a Time**: Isolate what works
3. **Use Version Control**: Git commit before major changes
4. **Test Thoroughly**: Run tests after modifications
5. **Monitor Training**: Use wandb to track changes

---

## 📚 Key Files Reference

| Component | File | What to Modify |
|-----------|------|----------------|
| **Model Architecture** | `nanochat/gpt.py` | Layers, attention, activations |
| **Optimizer** | `nanochat/adamw.py`, `nanochat/muon.py` | Learning rates, update rules |
| **Tokenizer** | `rustbpe/src/lib.rs` | BPE algorithm, vocab size |
| **Training Loop** | `scripts/base_train.py` | Training steps, loss, evaluation |
| **Data Loading** | `nanochat/dataloader.py` | Data preprocessing |
| **Inference** | `nanochat/engine.py` | Generation logic |

---

## 🚀 Next Steps

1. **Set up GPU in WSL2** (see `WSL_GPU_SETUP.md`)
2. **Use `train_1gpu.sh`** for real training
3. **Start customizing** one component at a time
4. **Experiment** and track results with wandb

Remember: WSL2 works great for GPU training - you don't need real Ubuntu! 🎉

