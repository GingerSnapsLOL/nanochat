# CPU/MPS demo run for Windows PowerShell
# Showing an example run for exercising some of the code paths on CPU
# Run as: powershell -ExecutionPolicy Bypass -File dev\runcpu.ps1
#
# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on CPU.
# Think of this run as educational/fun demo, not something you should expect to work well.

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "nanochat CPU Demo - Windows PowerShell" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "This is a small demo run for learning purposes." -ForegroundColor Yellow
Write-Host "For real training, use GPU with speedrun.ps1`n" -ForegroundColor Yellow

# Set environment variables
$env:OMP_NUM_THREADS = "1"
$env:NANOCHAT_BASE_DIR = "$env:USERPROFILE\.cache\nanochat"
New-Item -ItemType Directory -Force -Path $env:NANOCHAT_BASE_DIR | Out-Null

# Install uv if not present
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
}

# Setup Python environment
Write-Host "`n[1/5] Setting up Python environment..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    uv venv
}
Write-Host "Installing dependencies..." -ForegroundColor Yellow
uv sync --extra cpu
# Note: With uv, we use `uv run` instead of activating venv and using `python`

# Setup wandb
if (-not $env:WANDB_RUN) {
    $env:WANDB_RUN = "dummy"
}

# Install Rust if not present
Write-Host "`n[2/5] Setting up Rust..." -ForegroundColor Yellow
if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Rust is not installed!" -ForegroundColor Red
    Write-Host "Please install Rust from: https://rustup.rs/" -ForegroundColor Red
    exit 1
}

# Build tokenizer
Write-Host "Building Rust tokenizer..." -ForegroundColor Yellow
uv run maturin develop --release --manifest-path rustbpe\Cargo.toml

# Reset report
Write-Host "`n[3/5] Resetting report..." -ForegroundColor Yellow
uv run -m nanochat.report reset

# Train tokenizer on smaller dataset for CPU
Write-Host "`n[4/5] Training tokenizer (smaller dataset for CPU)..." -ForegroundColor Yellow
uv run -m nanochat.dataset -n 4
uv run -m scripts.tok_train --max_chars=1000000000
uv run -m scripts.tok_eval

# Base training with tiny model for CPU
Write-Host "`n[5/5] Training tiny model (this will be slow on CPU)..." -ForegroundColor Yellow
Write-Host "Training depth=4 model with very small batch size..." -ForegroundColor Gray
uv run -m scripts.base_train -- --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "CPU Demo Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nThis was just a learning exercise." -ForegroundColor Yellow
Write-Host "For real training, use GPU with: powershell -File speedrun.ps1" -ForegroundColor Yellow

