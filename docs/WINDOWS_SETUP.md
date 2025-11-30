# Windows Setup Guide for nanochat

This guide covers two options for running nanochat on Windows.

## Option 1: WSL2 (Ubuntu on Windows) - **RECOMMENDED** ⭐

This is the recommended approach. WSL2 provides a Linux environment that works seamlessly with nanochat.

### Installation Steps

1. **Install WSL2 and Ubuntu**
   ```powershell
   # Open PowerShell as Administrator and run:
   wsl --install
   ```
   This will install WSL2 and Ubuntu. Restart your computer when prompted.

2. **After restart, set up Ubuntu**
   - Create a username and password when prompted
   - Update packages:
     ```bash
     sudo apt update && sudo apt upgrade -y
     ```

3. **Install GPU Support (if you have NVIDIA GPU)**
   ```bash
   # Install NVIDIA drivers on Windows first from nvidia.com
   # Then in WSL, install CUDA toolkit:
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-4
   ```

4. **Clone and Setup nanochat in WSL**
   ```bash
   # Navigate to your Windows files (they're accessible from WSL)
   cd /mnt/c/Users/6204/Desktop/nanochat
   
   # Or clone fresh in WSL home directory:
   cd ~
   git clone https://github.com/karpathy/nanochat.git
   cd nanochat
   ```

5. **Run the scripts normally**
   ```bash
   # For CPU/MPS learning:
   bash dev/runcpu.sh
   
   # For full training (if you have GPU):
   bash speedrun.sh
   ```

### Accessing Windows Files from WSL
- Windows drives: `/mnt/c/`, `/mnt/d/`, etc.
- Your files: `/mnt/c/Users/6204/Desktop/nanochat`

### Accessing WSL Files from Windows
- WSL files: `\\wsl$\Ubuntu\home\yourusername\`

---

## Option 2: Native Windows (PowerShell Scripts)

If you prefer to stay in Windows PowerShell, use these converted scripts.

### Prerequisites
- Python 3.10+ installed
- Git for Windows
- PowerShell 5.1+ (comes with Windows)

### Setup Steps

1. **Install Rust**
   - Download from: https://rustup.rs/
   - Run the installer

2. **Install uv**
   ```powershell
   # In PowerShell:
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Run the PowerShell scripts** (see below)

---

## PowerShell Scripts

### speedrun.ps1

```powershell
# nanochat speedrun script for Windows PowerShell
# Run as: powershell -ExecutionPolicy Bypass -File speedrun.ps1

$ErrorActionPreference = "Stop"

# Set environment variables
$env:OMP_NUM_THREADS = "1"
$env:NANOCHAT_BASE_DIR = "$env:USERPROFILE\.cache\nanochat"
New-Item -ItemType Directory -Force -Path $env:NANOCHAT_BASE_DIR | Out-Null

# Install uv if not present
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
}

# Setup Python environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    uv venv
}
Write-Host "Installing dependencies..."
uv sync --extra gpu

# Activate venv
& ".venv\Scripts\Activate.ps1"

# Setup wandb
if (-not $env:WANDB_RUN) {
    $env:WANDB_RUN = "dummy"
}

# Install Rust if not present
if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Rust..."
    # You need to install Rust manually from https://rustup.rs/
    Write-Host "Please install Rust from https://rustup.rs/ and restart this script"
    exit 1
}

# Build tokenizer
Write-Host "Building Rust tokenizer..."
uv run maturin develop --release --manifest-path rustbpe\Cargo.toml

# Reset report
uv run -m nanochat.report reset

# Download dataset
Write-Host "Downloading dataset..."
uv run -m nanochat.dataset -n 8
Start-Job -ScriptBlock { uv run -m nanochat.dataset -n 240 } | Out-Null
$datasetJob = Get-Job

# Train tokenizer
Write-Host "Training tokenizer..."
uv run -m scripts.tok_train --max_chars=2000000000
uv run -m scripts.tok_eval

# Wait for dataset download
Write-Host "Waiting for dataset download..."
Wait-Job $datasetJob | Out-Null
Remove-Job $datasetJob

# Base training
Write-Host "Starting base training..."
$NPROC_PER_NODE = 1  # Adjust based on your GPU setup
if ($NPROC_PER_NODE -eq 1) {
    uv run -m scripts.base_train -- --depth=20 --run=$env:WANDB_RUN
} else {
    uv run uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --run=$env:WANDB_RUN
}

# Evaluate base model
uv run -m scripts.base_loss
uv run -m scripts.base_eval

# Download identity conversations
Write-Host "Downloading identity conversations..."
$identityPath = "$env:NANOCHAT_BASE_DIR\identity_conversations.jsonl"
Invoke-WebRequest -Uri "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl" -OutFile $identityPath

# Midtraining
Write-Host "Starting midtraining..."
if ($NPROC_PER_NODE -eq 1) {
    uv run -m scripts.mid_train -- --run=$env:WANDB_RUN
    uv run -m scripts.chat_eval -- -i mid
} else {
    uv run uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$env:WANDB_RUN
    uv run uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid
}

# SFT
Write-Host "Starting SFT..."
if ($NPROC_PER_NODE -eq 1) {
    uv run -m scripts.chat_sft -- --run=$env:WANDB_RUN
    uv run -m scripts.chat_eval -- -i sft
} else {
    uv run uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$env:WANDB_RUN
    uv run uv run torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft
}

# Generate report
uv run -m nanochat.report generate

Write-Host "Training complete! You can now chat with your model:"
Write-Host "uv run -m scripts.chat_web"
```

### runcpu.ps1 (for CPU learning)

```powershell
# CPU/MPS demo run for Windows PowerShell
# Run as: powershell -ExecutionPolicy Bypass -File dev\runcpu.ps1

$ErrorActionPreference = "Stop"

$env:OMP_NUM_THREADS = "1"
$env:NANOCHAT_BASE_DIR = "$env:USERPROFILE\.cache\nanochat"
New-Item -ItemType Directory -Force -Path $env:NANOCHAT_BASE_DIR | Out-Null

# Install uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
}

# Setup environment
if (-not (Test-Path ".venv")) {
    uv venv
}
uv sync --extra cpu
& ".venv\Scripts\Activate.ps1"

if (-not $env:WANDB_RUN) {
    $env:WANDB_RUN = "dummy"
}

# Install Rust
if (-not (Get-Command rustc -ErrorAction SilentlyContinue)) {
    Write-Host "Please install Rust from https://rustup.rs/"
    exit 1
}

# Build tokenizer
uv run maturin develop --release --manifest-path rustbpe\Cargo.toml

# Reset report
uv run -m nanochat.report reset

# Train tokenizer (smaller for CPU)
uv run -m nanochat.dataset -n 4
uv run -m scripts.tok_train --max_chars=1000000000

# Base training (tiny model for CPU)
uv run -m scripts.base_train -- --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20

Write-Host "CPU demo complete!"
```

---

## Comparison

| Feature | WSL2 | Native Windows |
|---------|------|----------------|
| **Ease of Setup** | ⭐⭐⭐ Easy | ⭐⭐ Moderate |
| **Script Compatibility** | ⭐⭐⭐ Perfect | ⭐ Needs conversion |
| **GPU Support** | ⭐⭐⭐ Full support | ⭐⭐ Limited |
| **Performance** | ⭐⭐⭐ Native-like | ⭐⭐ Good |
| **Documentation Match** | ⭐⭐⭐ Perfect | ⭐ Different paths |
| **Maintenance** | ⭐⭐⭐ Low | ⭐⭐ Higher |

## Recommendation

**Use WSL2** - It's the path of least resistance and matches the project's design. The setup is straightforward, and you'll have fewer compatibility issues.

---

## Troubleshooting

### WSL2 Issues

**WSL not starting:**
```powershell
# In PowerShell (Admin):
wsl --update
wsl --shutdown
```

**GPU not detected in WSL:**
- Make sure you have NVIDIA drivers installed on Windows
- Check: `nvidia-smi` in WSL should work

### Windows PowerShell Issues

**Execution Policy Error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Path Issues:**
- Use backslashes `\` in PowerShell
- Use forward slashes `/` in WSL

**Rust/Cargo Issues:**
- Make sure Rust is in your PATH
- Restart PowerShell after installing Rust

---

## Next Steps

1. **Choose your approach** (WSL2 recommended)
2. **Follow the setup steps** above
3. **Start with CPU demo** to learn: `bash dev/runcpu.sh` (WSL) or `powershell -File dev\runcpu.ps1` (Windows)
4. **Move to GPU training** when ready

Good luck! 🚀

