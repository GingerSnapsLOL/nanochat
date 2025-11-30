# Quick Reference: nanochat Training Pipeline

This is a quick reference guide for the nanochat training pipeline. Use this alongside the detailed LEARNING_PLAN.md.

## 🚀 Quick Start Commands

### Full Pipeline (4 hours, ~$100)
```bash
bash speedrun.sh
```

### CPU/MPS (for learning, much slower)
```bash
bash dev/runcpu.sh
```

### Individual Stages

#### 1. Tokenizer Training
```bash
# Download data
uv run -m nanochat.dataset -n 8

# Train tokenizer
uv run -m scripts.tok_train --max_chars=2000000000

# Evaluate tokenizer
uv run -m scripts.tok_eval
```

#### 2. Base Pretraining
```bash
# Single GPU
uv run -m scripts.base_train --depth=20

# Multi-GPU (8 GPUs)
uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# Evaluate base model
uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
```

#### 3. Midtraining
```bash
# Download identity conversations
curl -L -o ~/.cache/nanochat/identity_conversations.jsonl \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Train
uv run torchrun --standalone --nproc_per_node=8 -m scripts.mid_train

# Evaluate
uv run torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid
```

#### 4. Supervised Fine-Tuning (SFT)
```bash
# Train
uv run torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft

# Evaluate
uv run torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

#### 5. Inference
```bash
# CLI chat
uv run -m scripts.chat_cli

# Web UI
uv run -m scripts.chat_web
```

---

## 📁 Key Files Reference

### Core Model
- `nanochat/gpt.py` - GPT Transformer model
- `nanochat/tokenizer.py` - BPE tokenizer wrapper
- `nanochat/engine.py` - Inference engine with KV cache

### Training Scripts
- `scripts/base_train.py` - Base model pretraining
- `scripts/mid_train.py` - Midtraining (conversation format)
- `scripts/chat_sft.py` - Supervised fine-tuning
- `scripts/chat_rl.py` - Reinforcement learning (optional)

### Evaluation
- `scripts/base_eval.py` - CORE metric for base model
- `scripts/chat_eval.py` - Task evaluations (ARC, GSM8K, etc.)
- `scripts/base_loss.py` - Bits-per-byte evaluation

### Infrastructure
- `nanochat/dataloader.py` - Distributed data loading
- `nanochat/adamw.py` - AdamW optimizer
- `nanochat/muon.py` - Muon optimizer
- `nanochat/checkpoint_manager.py` - Save/load checkpoints

### Tasks
- `tasks/smoltalk.py` - General conversations
- `tasks/gsm8k.py` - Math problems
- `tasks/mmlu.py` - Multiple choice questions
- `tasks/humaneval.py` - Code generation
- `tasks/arc.py` - Science reasoning

---

## 🎛️ Key Hyperparameters

### Model Architecture
```python
depth = 20              # Number of transformer layers
max_seq_len = 2048      # Maximum context length
vocab_size = 65536      # Vocabulary size (from tokenizer)
```

### Training
```python
device_batch_size = 32          # Batch size per GPU
total_batch_size = 524288       # Effective batch size (with grad accum)
target_param_data_ratio = 20    # Chinchilla scaling (tokens = 20x params)

# Learning rates
embedding_lr = 0.2              # For embeddings
unembedding_lr = 0.004          # For output head
matrix_lr = 0.02                # For transformer layers (Muon)
```

### Scaling to Different Sizes
- **d20 (speedrun)**: ~561M params, ~11B tokens, ~4 hours, ~$100
- **d26**: Larger model, ~12 hours, ~$300
- **d32 (run1000.sh)**: ~1.9B params, ~38B tokens, ~33 hours, ~$800

---

## 📊 Training Stages Overview

```
┌─────────────────┐
│ 1. Tokenizer    │  Train BPE on ~2B chars, vocab_size=65536
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Base Train   │  Pretrain on raw text (Chinchilla scaling)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Midtraining  │  Teach conversation format, tool use, tasks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. SFT          │  Supervised fine-tuning for chat quality
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. RL (optional)│  Reinforcement learning (currently GSM8K only)
└────────┬────────┘
         │
         ▼
    Inference
```

---

## 🔍 Understanding Model Sizes

### Parameter Calculation
For a model with depth `d`:
- Approximate params: `~28M * d` (for d20: ~560M params)
- Training tokens: `params * 20` (Chinchilla scaling)
- Data shards needed: `(tokens * 4.8) / 250M` (chars per shard)

### Memory Considerations
- **OOM?** Reduce `device_batch_size` (32 → 16 → 8 → 4)
- **Too slow?** Increase batch size if memory allows
- **Less than 80GB VRAM?** Reduce batch size or model size

---

## 🐛 Common Issues & Solutions

### Out of Memory (OOM)
```bash
# Reduce batch size
--device_batch_size=16  # or 8, 4, 2, 1

# Reduce model size
--depth=16  # instead of 20

# Reduce sequence length
--max_seq_len=1024  # instead of 2048
```

### Slow Training
- Check data loading (should be fast)
- Ensure proper batch sizes
- Use multiple GPUs with `torchrun`
- Check if data download is complete

### Poor Model Quality
- Ensure enough training data (check shard count)
- Train for sufficient iterations
- Check all training stages completed
- Verify data quality

---

## 📈 Evaluation Metrics

### Base Model
- **CORE**: Composite metric for base model capability
- **Bits-per-byte (bpb)**: Lower is better (related to loss)

### Chat Model
- **ARC**: Science reasoning (Challenge/Easy)
- **GSM8K**: Math problem solving
- **HumanEval**: Code generation
- **MMLU**: Broad knowledge multiple choice
- **ChatCORE**: Conversational ability

---

## 🎨 Customization Quick Tips

### Add Custom Identity
1. Create `identity_conversations.jsonl` (see `dev/gen_synthetic_data.py`)
2. Download to `~/.cache/nanochat/`
3. Already included in midtraining pipeline

### Add Custom Task
1. Create new file in `tasks/` directory
2. Implement `Task` interface (see `tasks/common.py`)
3. Add to `TaskMixture` in `scripts/mid_train.py`

### Modify Model Architecture
1. Edit `nanochat/gpt.py`
2. Adjust `GPTConfig` parameters
3. Test with small model first

---

## 📝 Data Locations

- **Base directory**: `~/.cache/nanochat/` (or `$NANOCHAT_BASE_DIR`)
- **Data shards**: `~/.cache/nanochat/data_shards/`
- **Checkpoints**: `~/.cache/nanochat/checkpoints/`
- **Tokenizer**: `~/.cache/nanochat/tokenizer.model`

---

## 🔗 Useful Resources

- **GitHub Discussions**: For questions and guides
- **DeepWiki**: `https://deepwiki.com/karpathy/nanochat` (AI-powered code Q&A)
- **Packaged code**: Use `files-to-prompt` to ask LLMs about the codebase

---

## 💡 Pro Tips

1. **Start with CPU/MPS**: Use `dev/runcpu.sh` to learn without GPU costs
2. **Monitor with wandb**: Set `WANDB_RUN=your_name` before training
3. **Use screen**: For long training runs: `screen -S training bash speedrun.sh`
4. **Check report.md**: Generated after training with full metrics
5. **Read logs carefully**: They contain valuable debugging info

---

## 🎯 Training Checklist

Before starting:
- [ ] Environment set up (uv, Python, Rust)
- [ ] GPU access configured (if using GPU)
- [ ] Sufficient disk space (~50GB)
- [ ] wandb logged in (optional but recommended)

During training:
- [ ] Tokenizer trained successfully
- [ ] Data shards downloaded
- [ ] Base training completes
- [ ] Midtraining completes
- [ ] SFT completes
- [ ] Evaluations run

After training:
- [ ] Check `report.md` for metrics
- [ ] Test inference with `chat_cli` or `chat_web`
- [ ] Save checkpoints if needed
- [ ] Document your configuration

---

Happy training! 🚀

