"""
Finetune a base model to be a chat model.
Run on one GPU e.g. for debugging:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import load_model, load_checkpoint, find_latest_checkpoint
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# SFT Hyperparameters
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# input model options
source = "mid" # base|mid|sft , which checkpoint to load the model from (base model, midtrained model, or sft model). Ignored if continue_training=True
model_tag = None # model tag to load the model from (base model or midtrained model)
step = None # step to load the model from (base model or midtrained model)
# compute/precision
device_type = "" # cuda|cpu|mps (empty => autodetect)
dtype = "bfloat16"
device_batch_size = 4 # max to avoid OOM
# optimization
num_epochs = 1
num_iterations = -1 # override number of iterations (-1 = disable, use num_epochs to derive it)
target_examples_per_step = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.01 # initial learning rate as fraction of base learning rate (configurable via CLI: --init_lr_frac=0.01)
# evaluation and logging there of
eval_every = 100
eval_steps = 100
eval_metrics_every = 200
eval_metrics_max_problems = 1024
# checkpoint saving
save_every = 500 # save checkpoint every N iterations (0 = disable periodic saves, only save at end)
save_best = True # if True, save checkpoint whenever validation loss improves
continue_training = False # if True, resume from checkpoint (auto-detects model_tag if not set)
continue_from_best = False # if True and continue_training=True, resume from best checkpoint instead of latest
dry_run = 0 # dry_run=1 is for experiments: we will log to wandb but we won't write checkpoints or report
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # possibly useful for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=run, config=user_config, save_code=True)

# -----------------------------------------------------------------------------
# Checkpoint resuming logic (runs before model loading if continuing)
start_step = 0
min_val_loss = float("inf")  # Initialize, will be restored from checkpoint if resuming
saved_num_iterations = None  # Will be restored from checkpoint if resuming
resume_from_sft = False
resume_checkpoint_dir = None
resume_step = None
if continue_training:
    # When continuing, we load from SFT checkpoints, not from source
    checkpoints_dir = os.path.join(get_base_dir(), "chatsft_checkpoints")
    
    # Determine output_dirname (model tag for SFT checkpoints)
    # If model_tag is provided, use it; otherwise try to auto-detect
    output_dirname = model_tag
    if output_dirname is None:
        # Try to find the largest model in SFT checkpoints
        try:
            from nanochat.checkpoint_manager import find_largest_model
            output_dirname = find_largest_model(checkpoints_dir)
            print0(f"Auto-detected model_tag: {output_dirname}")
        except (FileNotFoundError, ValueError):
            print0("⚠️  Could not auto-detect model_tag from SFT checkpoints")
            print0("   Please specify --model_tag explicitly")
            continue_training = False
    
    if continue_training and output_dirname is not None:
        if continue_from_best:
            # Find best checkpoint (lowest validation loss)
            resume_checkpoint_dir, resume_step = find_latest_checkpoint(checkpoints_dir, model_tag=output_dirname)
            if resume_checkpoint_dir is not None:
                # Load all metadata files to find the best one
                import glob
                import json
                meta_files = glob.glob(os.path.join(resume_checkpoint_dir, "meta_*.json"))
                best_step = None
                best_val_loss = float("inf")
                for meta_file in meta_files:
                    try:
                        step = int(os.path.basename(meta_file).split("_")[-1].split(".")[0])
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta_data = json.load(f)
                        val_loss = meta_data.get("val_loss")
                        if val_loss is not None and val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_step = step
                    except (ValueError, KeyError, json.JSONDecodeError):
                        continue
                if best_step is not None:
                    resume_step = best_step
                    print0(f"🏆 Resuming from BEST checkpoint: {resume_checkpoint_dir} at step {resume_step} (val_loss: {best_val_loss:.6f})")
                else:
                    resume_checkpoint_dir = None
                    resume_step = None
            else:
                resume_step = None
        else:
            # Find latest checkpoint (highest step)
            resume_checkpoint_dir, resume_step = find_latest_checkpoint(checkpoints_dir, model_tag=output_dirname)
            if resume_checkpoint_dir is not None and resume_step is not None:
                print0(f"🔄 Resuming from LATEST checkpoint: {resume_checkpoint_dir} at step {resume_step}")
        
        if resume_checkpoint_dir is not None and resume_step is not None:
            resume_from_sft = True
            # We'll load the model from the checkpoint below
        else:
            print0(f"⚠️  --continue specified but no checkpoint found for {output_dirname}")
            print0("   Starting training from scratch...")
            continue_training = False
            resume_from_sft = False

# Load the model and tokenizer
if resume_from_sft:
    # Load from SFT checkpoint
    model, tokenizer, meta = load_model("sft", device, phase="train", model_tag=output_dirname, step=resume_step)
else:
    # Load from source (mid/base) or from SFT if source="sft"
    model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)

orig_model = model # original, uncompiled model
# model = torch.compile(model, dynamic=True) # doesn't work super well because of variable lengths of inputs
engine = Engine(model, tokenizer) # will be used for inline model evaluation only
# Get model_type from metadata (saved during base/mid training) or infer from config
model_type = meta.get("model_type", None)
if model_type is None:
    # Infer from config: alcoholic has extra fields
    if hasattr(model.config, "rope_theta") or hasattr(model.config, "intermediate_size"):
        model_type = "alcoholic"
    else:
        model_type = "gpt"
print0(f"Model type: {model_type}")
depth = model.config.n_layer

# -----------------------------------------------------------------------------
# Task data mixture we'll train on
identity_conversations_filepath = os.path.join(get_base_dir(), "identity_conversations.jsonl")
train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"), # 2.3K rows
    ARC(subset="ARC-Challenge", split="train"), # 1.1K rows
    GSM8K(subset="main", split="train"), # 8K rows
    SmolTalk(split="train", stop=10_000), # 10K rows of smoltalk
    CustomJSON(filepath=identity_conversations_filepath), # 1K rows of synthetic identity conversations
    SimpleSpelling(size=300, split="train"), # 300 rows of Simple Spelling (e.g. spell the word 'apple')
    SpellingBee(size=300, split="train"), # 300 rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
]) # 2.3K + 1.1K + 8K + 10K + 1K + 0.3K + 0.3K = 23K rows
val_ds = SmolTalk(split="test") # general conversations, 24K rows (though we don't actually use all of it)

# -----------------------------------------------------------------------------
# DataLoader

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>") # use <|assistant_end|> as the pad token is ok, these positions are masked in the loss
    # prepares a list of tokenized conversations into a batch and yields
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1 # seq of n creates inputs/targets of n-1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long) # -1 is ignore index
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            # recall -1 is the ignore index, so mask out targets where mask is 0
            row_targets = ids_tensor[1:]
            # mask[1:] omits the mask for the BOS token, which is never a target atm so it's ok
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1 # mask out targets where mask is 0
            targets[i, :n-1] = row_targets
        inputs = inputs.to(device) # move to device
        targets = targets.to(device)
        return inputs, targets
    # iterates over the dataset in epochs, tokenizes
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = device_batch_size * ddp_world_size
print0(f"Target examples per step: {target_examples_per_step}")
print0(f"Device batch size: {device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

if num_iterations == -1:
    # derive num_iterations from num_epochs and the size of the dataset
    assert num_epochs > 0, "num_epochs must be positive if num_iterations is -1"
    num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=device_batch_size)

# -----------------------------------------------------------------------------
# Initialize the Optimizer

optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
# Set the initial learning rate as a fraction of the base learning rate
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"] # save the initial learning so we can decay easily later

# -----------------------------------------------------------------------------
# Load optimizer states if resuming
if resume_from_sft:
    # Load optimizer states from checkpoint
    model_data, optimizer_data, meta_data = load_checkpoint(resume_checkpoint_dir, resume_step, device, load_optimizer=True)
    
    # Fix torch compile prefix if needed
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    
    # Load model weights (already loaded above, but ensure consistency)
    orig_model.load_state_dict(model_data, strict=True, assign=True)
    
    # Ensure model is in train mode after loading
    orig_model.train()
    model.train()
    
    # Load optimizer states
    if optimizer_data is not None and len(optimizer_data) == len(optimizers):
        for opt, opt_state in zip(optimizers, optimizer_data):
            opt.load_state_dict(opt_state)
        print0("✅ Loaded optimizer states")
    else:
        print0("⚠️  Optimizer state not found or mismatched, reinitializing optimizers")
    
    # Zero gradients after loading optimizer state to ensure clean state
    orig_model.zero_grad(set_to_none=True)
    
    # Get the saved step and adjust training
    start_step = resume_step + 1
    
    # Restore num_iterations from checkpoint if available
    saved_num_iterations = meta_data.get("user_config", {}).get("num_iterations", num_iterations)
    if saved_num_iterations is not None and saved_num_iterations > 0:
        print0(f"📊 Original target was {saved_num_iterations} steps, remaining: {saved_num_iterations - start_step} steps")
    
    # Restore min_val_loss from checkpoint if available
    saved_min_val_loss = meta_data.get("min_val_loss", None)
    if saved_min_val_loss is not None:
        min_val_loss = saved_min_val_loss
        print0(f"📊 Restored min validation loss: {min_val_loss:.6f}")
    else:
        # Fallback to val_loss if min_val_loss not available
        saved_val_loss = meta_data.get("val_loss", None)
        if saved_val_loss is not None:
            min_val_loss = saved_val_loss
            print0(f"📊 Restored validation loss: {min_val_loss:.6f}")
    
    print0(f"📊 Resuming from step {start_step} (checkpoint was at step {resume_step})")

# -----------------------------------------------------------------------------
# Training loop

# Learning rate scheduler
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm

# Go!
step = start_step
train_iter = iter(train_loader)
# Skip forward in the iterator if resuming (approximate, since it's an infinite loop)
# Note: This is a best-effort approach. For exact resuming, we'd need to track dataset position.
# For SFT, this is usually acceptable since training is typically short.
for _ in range(start_step):
    try:
        next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        break

# Initialize metrics dict to avoid NameError
metrics = {}

for step in range(start_step, num_iterations):
    last_step = step == num_iterations - 1

    # evaluate the validation loss
    val_loss = None
    improved = False
    if last_step or step % eval_every == 0:
        model.eval()
        val_iter = iter(build_val_loader())
        losses = []
        for _ in range(eval_steps):
            val_inputs, val_targets = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean() # average over eval_steps
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG) # average over ranks
        val_loss = val_loss.item()
        improved = val_loss < min_val_loss
        if improved:
            min_val_loss = val_loss
            if save_best and master_process and not dry_run:
                print0(f"🎯 New best validation loss: {min_val_loss:.6f} - saving best model")
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        wandb_run.log({
            "step": step,
            "val_loss": val_loss,
        })
        model.train()
    else:
        # Use previous val_loss if available, otherwise use a placeholder
        val_loss = min_val_loss if min_val_loss != float("inf") else None

    # evlauate accuracy of the multiple choice tasks (which are quick to run)
    if last_step or (step > 0 and step % eval_metrics_every == 0):
        model.eval()
        metrics = {}
        with torch.no_grad(), autocast_ctx:
            # note that because these are inside no_grad, we can usually afford to at least ~2X the batch size
            metrics["mmlu_acc"] = run_chat_eval("MMLU", model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=eval_metrics_max_problems)
            metrics["arc_easy_acc"] = run_chat_eval("ARC-Easy", model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=eval_metrics_max_problems)
        metrics_str = ', '.join(f'{k}: {v:.6f}' for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        wandb_run.log({
            "step": step,
            **metrics,
        })
        model.train()

    # save checkpoint periodically, on best model, and at the end (only on master process)
    should_save = False
    save_reason = ""
    if master_process and not dry_run:
        if last_step:
            should_save = True
            save_reason = "final"
        elif save_every > 0 and step > 0 and step % save_every == 0:
            should_save = True
            save_reason = "periodic"
        elif save_best and improved:  # Save when validation improved
            should_save = True
            save_reason = "best"
    
    if should_save:
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = f"{model_type}_d{depth}" # base the model tag on model type and depth, e.g. gpt_d12 or alcoholic_d12
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag)
        model_config_kwargs = model.config.__dict__ # works for both GPTConfig and AlcoholicNanoConfig
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers], # Save optimizer states for resuming
            {
                "step": step,
                "val_loss": val_loss if val_loss is not None else min_val_loss, # validation loss at this step
                "min_val_loss": min_val_loss, # best validation loss so far
                "model_type": model_type, # save which model type was used
                **metrics, # metrics if evaluated (empty dict if not evaluated yet)
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
            }
        )
        print0(f"✅ Saved checkpoint at step {step:05d} ({save_reason})")

    if last_step:
        break

    # evaluate the gradient
    num_tokens = torch.tensor(0, device=device) # the number of "active" tokens of supervision seen
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward() # accumulate the gradient
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM) # sum over ranks

    # learning rate scheduler
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # step the optimizers
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    # logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    print0(f"Step {step:05d}/{num_iterations:05d} | Training loss: {train_loss_item:.6f}| lrm: {lrm:.6f}| num_tokens: {num_tokens_item:,}")
    wandb_run.log({
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
    })
    step += 1

# Final checkpoint save is handled in the loop above when last_step is True

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat SFT", data=[
    user_config, # CLI args
    {
        "Training rows": len(train_ds),
        "Number of iterations": num_iterations,
        "Training loss": train_loss_item,
        "Validation loss": val_loss,
    },
])

# Cleanup
wandb_run.finish()
compute_cleanup()
