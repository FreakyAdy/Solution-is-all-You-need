# PHANTOM Installation Guide

## Quickstart

### Linux / macOS / WSL2
```bash
curl -fsSL https://phantom-core.org/install.sh | bash
```

### Windows (PowerShell)
```powershell
irm https://phantom-core.org/install.ps1 | iex
```

## Manual Installation

### 1. Prerequisites
- NVIDIA GPU with Pascal or newer architecture (SM 6.0+)
- CUDA Toolkit 12.0+
- Python 3.10+
- Rust 1.75+ (for core engine compilation)

### 2. Python Environment Setup
```bash
git clone https://github.com/FreakyAdy/Solution-is-all-You-need.git
cd Solution-is-all-You-need
pip install -e python/
```

### 3. Build Rust Core Engine
```bash
cargo build --release
```

### 4. Verify Installation
```bash
phantom doctor
phantom plan llama3:70b
```
