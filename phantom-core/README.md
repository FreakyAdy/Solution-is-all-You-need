# PHANTOM CORE — Quickstart Guide

This project contains the complete PHANTOM CORE architecture as specified in the Master Prompt.

## Project Structure

```
phantom-core/
├── kernels/                  # Phase 1: Custom CUDA Kernels
│   ├── common/               # Shared headers (FP8 E4M3, timing)
│   ├── spectral_quant/       # 1D DCT, top-K selection, Fused iDCT-GEMV
│   ├── neural_cache/         # KV Autoencoder (Encode/Decode + Training)
│   ├── sparse_moe/           # Gate prediction and Sparse Matmul
│   └── attention/            # FlashAttention-3 and GQA
├── core/                     # Phase 2 & 4: Rust Engine
│   ├── src/
│   │   ├── memory/           # VRAM manager, CPU Offload, NVMe Phantom Pages, LRU Map
│   │   ├── engine/           # Orchestration engine linking CUDA to memory
│   │   ├── sampler/          # Resonance Sampler (Innovation 6)
│   │   ├── scheduler/        # Chronos Scheduler (Innovation 7)
│   │   ├── ipc/              # Inter-Process Communication
│   │   ├── lib.rs            # Library entry points & Types
│   │   └── main.rs           # CLI Entry Point
│   └── Cargo.toml            # Rust Build configuration
└── python_api/               # Phase 3: Python Wrapper & Calibration
    ├── setup.py              # Packaging
    └── phantom/
        ├── api/              # FastAPI Server (OpenAI compatible)
        └── calibration/      # Calibration Pipeline (generates .phantom profiles)
```

## How to Build

### 1. Build the CUDA Kernels
Requires CUDA Toolkit 12.0+.
```bash
cd kernels
mkdir build && cd build
cmake ..
make -j
```

### 2. Build the Rust Core
Requires Rust toolchain.
```bash
cd core
cargo build --release
```

### 3. Install Python API
```bash
cd python_api
pip install -e .
```

## Usage Example

1. **Calibrate a Model:**
   ```bash
   phantom-core calibrate --model meta-llama/Meta-Llama-3-70B --output ~/profiles/llama70b
   ```

2. **Serve the Model:**
   ```bash
   phantom-core serve --model meta-llama/Meta-Llama-3-70B --profile ~/profiles/llama70b/profile.phantom
   ```

3. **Query the API:**
   ```bash
   curl http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Explain hardware-transcendent computing.",
       "max_tokens": 100
     }'
   ```
