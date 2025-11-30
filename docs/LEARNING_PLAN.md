# Learning Plan: Building Your Own LLM with nanochat

This is a comprehensive learning plan to help you understand and build your own LLM using the nanochat repository. nanochat is a full-stack implementation of a ChatGPT-like model, covering everything from tokenization to deployment.

## 📚 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Phase 1: Understanding the Architecture](#phase-1-understanding-the-architecture)
3. [Phase 2: Core Components Deep Dive](#phase-2-core-components-deep-dive)
4. [Phase 3: Training Pipeline](#phase-3-training-pipeline)
5. [Phase 4: Hands-On Practice](#phase-4-hands-on-practice)
6. [Phase 5: Customization & Advanced Topics](#phase-5-customization--advanced-topics)
7. [Resources & Next Steps](#resources--next-steps)

---

## Prerequisites

### Essential Knowledge
- **Python**: Strong understanding of Python programming
- **PyTorch**: Familiarity with neural networks, tensors, and basic training loops
- **Deep Learning Fundamentals**: 
  - Understanding of neural networks, backpropagation
  - Basic knowledge of Transformers (attention mechanism, positional encoding)
  - Understanding of training concepts (loss, optimization, learning rates)

### Recommended Knowledge
- **Distributed Training**: Basic understanding of multi-GPU training (helpful but not required)
- **Natural Language Processing**: Basic understanding of tokenization, language modeling
- **Git**: Basic version control

### Setup Requirements
- Python 3.10+
- Access to GPU (recommended: 8xH100 or similar, but can start with CPU/MPS for learning)
- ~50GB disk space for data and checkpoints
- Basic familiarity with command line

---

## Phase 1: Understanding the Architecture

**Goal**: Understand the overall structure and flow of nanochat

### Week 1: Repository Overview

#### Day 1-2: Read and Understand
1. **Read the README.md thoroughly**
   - Understand what nanochat is and its goals
   - Note the file structure and what each directory contains
   - Understand the training pipeline stages

2. **Explore the codebase structure**
   - Navigate through `nanochat/` directory (core modules)
   - Navigate through `scripts/` directory (training scripts)
   - Navigate through `tasks/` directory (evaluation tasks)

#### Day 3-4: Run the Speedrun
1. **Set up environment**
   ```bash
   # Follow the setup instructions in README
   # Install dependencies, set up GPU access if available
   ```

2. **Run a small test** (if on CPU/MPS, use `dev/runcpu.sh` as reference)
   - Start with a tiny model to understand the flow
   - Watch the training logs to see what happens at each stage

3. **Study `speedrun.sh`**
   - Understand the sequence of operations
   - Identify each training stage
   - Note where data is downloaded and processed

#### Day 5-7: High-Level Architecture
1. **Study the training pipeline flow**
   - Tokenizer training → Base pretraining → Midtraining → SFT → RL (optional)
   - Understand why each stage exists
   - Map out the data flow between stages

2. **Key Questions to Answer**:
   - What is the difference between base training and midtraining?
   - Why do we need SFT after midtraining?
   - What does each evaluation script measure?

---

## Phase 2: Core Components Deep Dive

**Goal**: Understand each major component in detail

### Week 2: Model Architecture

#### Day 1-3: The GPT Model (`nanochat/gpt.py`)
1. **Read and understand**:
   - `GPTConfig` dataclass - what parameters define the model
   - `GPT` class - the main model
   - `CausalSelfAttention` - how attention works
   - `Block` - transformer block structure

2. **Key concepts to understand**:
   - **Rotary Embeddings (RoPE)**: How positional encoding works
   - **QK Norm**: Why we normalize queries and keys
   - **Multi-Query Attention (MQA)**: Efficiency optimization
   - **KV Cache**: How inference is optimized
   - **RMSNorm**: The normalization used

3. **Exercises**:
   - Draw a diagram of the model architecture
   - Trace through a forward pass with a small example
   - Understand how `generate()` works for inference

#### Day 4-5: Tokenizer (`nanochat/tokenizer.py`, `rustbpe/`)
1. **Understand BPE (Byte Pair Encoding)**:
   - Why we need tokenization
   - How BPE works conceptually
   - Why it's implemented in Rust (performance)

2. **Study the tokenizer interface**:
   - How to encode/decode text
   - Special tokens and their purpose
   - Vocabulary size and its impact

3. **Exercise**: 
   - Tokenize some sample text
   - Understand the relationship between characters and tokens

#### Day 6-7: Data Loading (`nanochat/dataloader.py`, `nanochat/dataset.py`)
1. **Understand**:
   - How pretraining data is downloaded and organized
   - How data is loaded and tokenized on-the-fly
   - Distributed data loading for multi-GPU training

2. **Study**:
   - `tokenizing_distributed_data_loader` function
   - How data shards work
   - How sequences are created from raw text

### Week 3: Training Infrastructure

#### Day 1-2: Optimizers (`nanochat/adamw.py`, `nanochat/muon.py`)
1. **Understand**:
   - Why different optimizers for different parts (AdamW vs Muon)
   - How `setup_optimizers()` works
   - Learning rate scheduling

2. **Key concepts**:
   - Why embeddings have different learning rates
   - What Muon optimizer is and why it's used
   - Gradient accumulation for large batch sizes

#### Day 3-4: Checkpointing (`nanochat/checkpoint_manager.py`)
1. **Understand**:
   - How models are saved and loaded
   - What information is stored in checkpoints
   - How to resume training

#### Day 5-7: Inference (`nanochat/engine.py`)
1. **Study**:
   - How KV cache speeds up inference
   - How generation works step-by-step
   - Temperature, top-k sampling

2. **Exercise**:
   - Write a simple inference script
   - Generate text from a checkpoint

---

## Phase 3: Training Pipeline

**Goal**: Understand each training stage in detail

### Week 4: Stage 1 - Tokenizer Training

#### Day 1-3: Study `scripts/tok_train.py`
1. **Understand**:
   - How BPE tokenizer is trained
   - What data is used for training
   - Vocabulary size selection

2. **Key questions**:
   - Why train on ~2B characters?
   - How does vocabulary size affect model performance?
   - What is the compression ratio?

#### Day 4-5: Study `scripts/tok_eval.py`
1. **Understand evaluation metrics**:
   - Compression ratio
   - Tokens per character
   - How to interpret these metrics

### Week 5: Stage 2 - Base Pretraining

#### Day 1-4: Study `scripts/base_train.py`
1. **Understand the training loop**:
   - Forward pass
   - Loss computation
   - Backward pass and gradient accumulation
   - Optimizer step
   - Learning rate scheduling

2. **Key hyperparameters**:
   - `depth`: Number of transformer layers
   - `device_batch_size`: Batch size per GPU
   - `total_batch_size`: Effective batch size (with gradient accumulation)
   - `target_param_data_ratio`: Chinchilla scaling (20x params)
   - Learning rates for different components

3. **Training dynamics**:
   - How loss decreases over time
   - Validation bits-per-byte (bpb) metric
   - CORE metric evaluation
   - Sampling during training

4. **Exercises**:
   - Modify hyperparameters and observe effects
   - Add custom logging
   - Understand gradient clipping

#### Day 5-7: Evaluation (`scripts/base_eval.py`, `scripts/base_loss.py`)
1. **Understand**:
   - CORE metric: What it measures and why
   - Bits-per-byte: How it relates to loss
   - Sampling: How to generate text from base model

### Week 6: Stage 3 - Midtraining

#### Day 1-4: Study `scripts/mid_train.py`
1. **Understand the purpose**:
   - Why midtraining exists (teach conversation format)
   - What tasks are included (SmolTalk, MMLU, GSM8K, etc.)
   - How it differs from base training

2. **Study the task system** (`tasks/` directory):
   - `TaskMixture`: How multiple tasks are combined
   - Individual tasks: What each teaches the model
   - Data format: How conversations are structured

3. **Key concepts**:
   - Special tokens (`<|user|>`, `<|assistant|>`, etc.)
   - Tool use (calculator execution)
   - Multiple choice formatting

#### Day 5-7: Evaluation (`scripts/chat_eval.py`)
1. **Understand evaluation tasks**:
   - ARC: Science reasoning
   - GSM8K: Math problems
   - HumanEval: Code generation
   - MMLU: Broad knowledge
   - ChatCORE: Conversational ability

### Week 7: Stage 4 - Supervised Fine-Tuning (SFT)

#### Day 1-4: Study `scripts/chat_sft.py`
1. **Understand**:
   - What SFT does (domain adaptation)
   - How it differs from midtraining
   - Why it improves performance

2. **Data format**:
   - How conversations are formatted
   - Loss masking (only compute loss on assistant tokens)

3. **Training specifics**:
   - Usually shorter training than midtraining
   - Lower learning rates
   - Focus on conversation quality

### Week 8: Stage 5 - Reinforcement Learning (Optional)

#### Day 1-3: Study `scripts/chat_rl.py`
1. **Understand**:
   - What RL does (reward-based learning)
   - Currently only on GSM8K
   - How rewards are computed

2. **Note**: This is optional and more advanced

---

## Phase 4: Hands-On Practice

**Goal**: Build confidence by running and modifying the code

### Week 9: Small-Scale Training

#### Day 1-2: CPU/MPS Training
1. **Modify `dev/runcpu.sh`** or create your own:
   - Train a tiny model (depth=4, small batch size)
   - Understand memory constraints
   - Observe training dynamics

2. **Experiment**:
   - Change model depth
   - Adjust batch sizes
   - Modify learning rates
   - Observe effects on training

#### Day 3-5: Understanding Scaling
1. **Study scaling laws**:
   - Chinchilla scaling (20x params in tokens)
   - How model size affects performance
   - Compute budget considerations

2. **Experiment with different sizes**:
   - d4, d8, d12, d20 models
   - Compare training time and performance

#### Day 6-7: Debugging and Monitoring
1. **Learn to debug**:
   - Common errors (OOM, NaN, etc.)
   - How to read training logs
   - Using wandb for visualization

### Week 10: Customization

#### Day 1-3: Custom Data
1. **Add custom training data**:
   - Study `dev/gen_synthetic_data.py`
   - Create your own identity conversations
   - Add to midtraining pipeline

2. **Exercise**: 
   - Create a custom task in `tasks/`
   - Add it to the midtraining mixture

#### Day 4-5: Model Modifications
1. **Experiment with architecture**:
   - Change activation functions
   - Modify attention mechanisms
   - Adjust normalization

2. **Note**: Start small, test thoroughly

#### Day 6-7: Evaluation
1. **Add custom evaluation**:
   - Create new tasks
   - Measure specific capabilities
   - Compare before/after modifications

---

## Phase 5: Customization & Advanced Topics

**Goal**: Make nanochat your own

### Week 11: Advanced Customization

#### Day 1-3: Personality and Identity
1. **Study identity infusion**:
   - Read the guide in Discussions: "infusing identity to your nanochat"
   - Understand synthetic data generation
   - Create your own personality

2. **Exercise**:
   - Generate custom identity conversations
   - Train a model with your personality
   - Compare outputs

#### Day 4-5: Adding Capabilities
1. **Study ability addition**:
   - Read: "counting r in strawberry (and how to add abilities generally)"
   - Understand how to teach new skills
   - Create a new ability task

#### Day 6-7: Deployment
1. **Study web interface**:
   - `scripts/chat_web.py` - how the web UI works
   - `nanochat/ui.html` - frontend
   - How to serve your model

### Week 12: Production Considerations

#### Day 1-3: Optimization
1. **Study**:
   - Model compilation (`torch.compile`)
   - Quantization (if implemented)
   - Inference optimization

2. **Experiment**:
   - Measure inference speed
   - Optimize for your use case

#### Day 4-5: Monitoring and Logging
1. **Set up proper logging**:
   - wandb integration
   - Custom metrics
   - Training dashboards

#### Day 6-7: Documentation
1. **Document your changes**:
   - Keep notes on experiments
   - Document hyperparameters
   - Create your own guide

---

## Resources & Next Steps

### Essential Reading
1. **nanochat Discussions**: 
   - "Introducing nanochat: The best ChatGPT that $100 can buy"
   - "Guide: infusing identity to your nanochat"
   - "Guide: counting r in strawberry"

2. **Related Papers** (for deeper understanding):
   - Attention Is All You Need (Transformer architecture)
   - GPT papers (GPT-1, GPT-2, GPT-3)
   - Chinchilla (scaling laws)
   - InstructGPT (SFT and RLHF)

3. **Code References**:
   - nanoGPT (Karpathy's earlier project)
   - HuggingFace Transformers (for comparison)

### Practice Projects

#### Beginner
1. Train a tiny model (d4) on CPU
2. Modify the model's name/identity
3. Add a simple custom task

#### Intermediate
1. Train a d20 model on GPU
2. Create a custom evaluation task
3. Fine-tune for a specific domain

#### Advanced
1. Scale to d26 or d32
2. Implement a new optimization technique
3. Add new capabilities (tool use, etc.)
4. Create a production deployment

### Community
- GitHub Discussions: Ask questions, share experiments
- Issues: Report bugs, suggest improvements
- Pull Requests: Contribute back to the project

### Key Metrics to Track
- **Training**: Loss, validation bpb, CORE metric
- **Evaluation**: Task-specific metrics (ARC, GSM8K, etc.)
- **Inference**: Speed, quality of generations
- **Cost**: Training time, GPU hours, total cost

### Common Pitfalls to Avoid
1. **OOM (Out of Memory)**: Reduce batch size, model size, or sequence length
2. **Overfitting**: Monitor validation metrics, use proper data splits
3. **Underfitting**: Train longer, check data quality
4. **Slow training**: Check data loading, use proper batch sizes
5. **Poor quality**: Ensure enough data, proper training stages

---

## Learning Checklist

Use this to track your progress:

### Phase 1: Architecture
- [ ] Read and understood README
- [ ] Explored codebase structure
- [ ] Ran speedrun or small test
- [ ] Understand training pipeline flow

### Phase 2: Components
- [ ] Understand GPT model architecture
- [ ] Understand tokenizer
- [ ] Understand data loading
- [ ] Understand optimizers
- [ ] Understand inference

### Phase 3: Training
- [ ] Understand tokenizer training
- [ ] Understand base pretraining
- [ ] Understand midtraining
- [ ] Understand SFT
- [ ] Understand RL (optional)

### Phase 4: Practice
- [ ] Trained a small model
- [ ] Modified hyperparameters
- [ ] Added custom data
- [ ] Created custom task
- [ ] Evaluated model

### Phase 5: Advanced
- [ ] Customized model personality
- [ ] Added new capabilities
- [ ] Deployed model
- [ ] Optimized for production

---

## Tips for Success

1. **Start Small**: Begin with tiny models to understand the flow
2. **Read Code**: Don't just run scripts - read and understand them
3. **Experiment**: Modify parameters and observe effects
4. **Take Notes**: Document what you learn and what works
5. **Ask Questions**: Use GitHub Discussions for help
6. **Be Patient**: Training takes time, especially for larger models
7. **Iterate**: Each experiment teaches you something new

---

## Estimated Timeline

- **Fast Track (Full-time)**: 6-8 weeks
- **Part-time (10-15 hrs/week)**: 12-16 weeks
- **Casual (5 hrs/week)**: 20-24 weeks

Remember: The goal is understanding, not speed. Take your time to truly understand each component.

---

Good luck on your journey to building your own LLM! 🚀

