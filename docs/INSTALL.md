# PHANTOM Installation Guide

## Quickstart
 
```bash
git clone https://github.com/FreakyAdy/phantom.git
cd phantom
pip install -e python/
```

## Manual Installation

### 1. Prerequisites
- NVIDIA GPU with Pascal or newer architecture (SM 6.0+)
- CUDA Toolkit 12.0+
- Python 3.10+
- Rust 1.75+ (for core engine compilation)

### 2. Python Environment Setup

PHANTOM auto-uses your GPU when a CUDA-enabled PyTorch is installed and falls back to CPU otherwise. The default `pip install torch` is CPU-only on Windows/macOS, so install the CUDA build first:

```bash
# Requires CUDA Toolkit + driver, NVIDIA GPU (Pascal+)
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

```powershell
# Windows PowerShell (Python 3.14 example)
pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

> Pick the CUDA wheel matching your Python version at https://pytorch.org/get-started/locally. CPU-only torch is also supported — the runtime just runs on the CPU SIMD engine.

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
