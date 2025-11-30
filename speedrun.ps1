# nanochat speedrun script for Windows PowerShell
# This script is the "Best ChatGPT clone that $100 can buy"
# Run as: powershell -ExecutionPolicy Bypass -File speedrun.ps1
# Or with wandb: $env:WANDB_RUN="speedrun"; powershell -ExecutionPolicy Bypass -File speedrun.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "nanochat Speedrun - Windows PowerShell" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Set environment variables
$env:OMP_NUM_THREADS = "1"
$env:NANOCHAT_BASE_DIR = "$env:USERPROFILE\.cache\nanochat"
New-Item -ItemType Directory -Force -Path $env:NANOCHAT_BASE_DIR | Out-Null
Write-Host "Base directory: $env:NANOCHAT_BASE_DIR" -ForegroundColor Green

# -----------------------------------------------------------------------------
# Python venv setup with uv

Write-Host "`n[1/8] Setting up Python environment..." -ForegroundColor Yellow

# Install uv if not present
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # Add to PATH for current session
    $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
}

# Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    uv venv
}

# Install dependencies
Write-Host "Installing dependencies (this may take a while)..." -ForegroundColor Yellow
uv sync --extra gpu
# Note: With uv, we use `uv run` instead of activating venv and using `python`

# -----------------------------------------------------------------------------
# wandb setup

if (-not $env:WANDB_RUN) {
    $env:WANDB_RUN = "dummy"
    Write-Host "Using dummy wandb (no logging). Set `$env:WANDB_RUN to enable." -ForegroundColor Gray
}

# -----------------------------------------------------------------------------
# Reset report

Write-Host "`n[2/8] Resetting report..." -ForegroundColor Yellow
uv run -m nanochat.report reset

# -----------------------------------------------------------------------------
# Install Rust / Cargo

Write-Host "`n[3/8] Setting up Rust..." -ForegroundColor Yellow

if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Rust is not installed!" -ForegroundColor Red
    Write-Host "Please install Rust from: https://rustup.rs/" -ForegroundColor Red
    Write-Host "After installing, restart PowerShell and run this script again." -ForegroundColor Red
    exit 1
}

# Build the rustbpe Tokenizer
Write-Host "Building Rust tokenizer..." -ForegroundColor Yellow
uv run maturin develop --release --manifest-path rustbpe\Cargo.toml

# -----------------------------------------------------------------------------
# Tokenizer

Write-Host "`n[4/8] Training tokenizer..." -ForegroundColor Yellow

# Download the first ~2B characters of pretraining dataset
Write-Host "Downloading dataset shards (this may take a while)..." -ForegroundColor Yellow
uv run -m nanochat.dataset -n 8

# Start downloading more shards in background
Write-Host "Starting background download of additional shards..." -ForegroundColor Yellow
$datasetJob = Start-Job -ScriptBlock {
    $env:Path = $using:env:Path
    Set-Location $using:PWD
    uv run -m nanochat.dataset -n 240
}

# Train the tokenizer
Write-Host "Training tokenizer on ~2B characters..." -ForegroundColor Yellow
uv run -m scripts.tok_train --max_chars=2000000000

# Evaluate the tokenizer
uv run -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

Write-Host "`n[5/8] Base model pretraining..." -ForegroundColor Yellow

# Wait for dataset download to complete
Write-Host "Waiting for dataset download to complete..." -ForegroundColor Yellow
Wait-Job $datasetJob | Out-Null
Receive-Job $datasetJob | Out-Null
Remove-Job $datasetJob
Write-Host "Dataset download complete!" -ForegroundColor Green

# Number of processes/GPUs to use (adjust based on your setup)
$NPROC_PER_NODE = 1  # Change to 8 if you have 8 GPUs

if ($NPROC_PER_NODE -eq 1) {
    Write-Host "Starting single-GPU pretraining..." -ForegroundColor Yellow
    uv run -m scripts.base_train -- --depth=20 --run=$env:WANDB_RUN
} else {
    Write-Host "Starting multi-GPU pretraining ($NPROC_PER_NODE GPUs)..." -ForegroundColor Yellow
    uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --run=$env:WANDB_RUN
}

# Evaluate the model
Write-Host "Evaluating base model..." -ForegroundColor Yellow
if ($NPROC_PER_NODE -eq 1) {
    uv run -m scripts.base_loss
    uv run -m scripts.base_eval
} else {
    uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
    uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval
}

# -----------------------------------------------------------------------------
# Midtraining

Write-Host "`n[6/8] Midtraining..." -ForegroundColor Yellow

# Download identity conversations
Write-Host "Downloading identity conversations..." -ForegroundColor Yellow
$identityPath = "$env:NANOCHAT_BASE_DIR\identity_conversations.jsonl"
Invoke-WebRequest -Uri "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl" -OutFile $identityPath

# Run midtraining
if ($NPROC_PER_NODE -eq 1) {
    uv run -m scripts.mid_train -- --run=$env:WANDB_RUN
    uv run -m scripts.chat_eval -- -i mid
} else {
    uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$env:WANDB_RUN
    uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid
}

# -----------------------------------------------------------------------------
# Supervised Finetuning

Write-Host "`n[7/8] Supervised Fine-Tuning (SFT)..." -ForegroundColor Yellow

if ($NPROC_PER_NODE -eq 1) {
    uv run -m scripts.chat_sft -- --run=$env:WANDB_RUN
    uv run -m scripts.chat_eval -- -i sft
} else {
    uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$env:WANDB_RUN
    uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft
}

# -----------------------------------------------------------------------------
# Generate report

Write-Host "`n[8/8] Generating report..." -ForegroundColor Yellow
uv run -m nanochat.report generate

# -----------------------------------------------------------------------------
# Done!

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nYou can now chat with your model:" -ForegroundColor Cyan
Write-Host "  uv run -m scripts.chat_cli" -ForegroundColor White
Write-Host "  uv run -m scripts.chat_web" -ForegroundColor White
Write-Host "`nReport saved to: report.md" -ForegroundColor Cyan

