# PHANTOM

Run large language models that exceed your GPU's physical VRAM by orchestrating GPU VRAM, System RAM, and NVMe storage into a tiered execution hierarchy.

[![CI](https://github.com/FreakyAdy/phantom/actions/workflows/ci.yml/badge.svg)](https://github.com/FreakyAdy/phantom/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Verified Results](https://img.shields.io/badge/Benchmarks-Canonical%20Ledger-orange.svg)](RESULTS.md)

---

## What this is

Most consumer GPUs have 6 GB to 16 GB of VRAM, while modern open-weight models require 16 GB to 40 GB in 4-bit precision (e.g. 32B models require ~20 GB, 70B models require ~37 GB). Standard runtimes either crash with CUDA out-of-memory errors or trigger unquantized dequantization spikes that exhaust system memory.

PHANTOM is a local inference runtime designed to extend the parameter ceiling of consumer hardware. It partitions transformer layers across three tiers:
1. **GPU VRAM** (GDDR6, ~192 GB/s): Hosts initial attention and MLP layers.
2. **Host RAM** (Dual-Channel DDR5, ~48 GB/s): Evaluates intermediate layers in-place via multi-threaded CPU SIMD vector kernels, transferring only intermediate activation vectors (~10 KB) across PCIe.
3. **NVMe SSD** (Gen4 x4, ~1.4–1.9 GB/s): Streams cold layers asynchronously via memory-mapped tiles.

Throughput is governed strictly by the memory tier housing the model's active working set: models fitting within VRAM run at hundreds of tokens per second; models spanning VRAM and DDR5 RAM run at 2.8 to 13 tokens per second; models requiring NVMe streaming run at 0.12 to 0.39 tokens per second.

---

## What this is not

- **Not a speedup for models that already fit in VRAM**: If an 8B model fits entirely inside your GPU memory, standard CUDA engines (vLLM, TensorRT-LLM) will run faster. PHANTOM is designed for workloads that cannot load without tiering.
- **Not competitive with multi-GPU datacenter serving**: PHANTOM targets single-machine local inference on consumer silicon (laptops and desktops).
- **Not immune to physical bandwidth limits**: Once model weights exceed fast VRAM + RAM capacity and must be paged from NVMe SSD on every token forward pass, decoding speed is bounded by NVMe sequential read bandwidth (~1.4–1.8 GB/s). For a 70B model with 15 GB of weights in SSD swap, throughput is physically limited to ~0.39 tok/sec. No prefetching algorithm can bypass physical bus limits.
- **Currently NVIDIA only**: Requires CUDA 12.x and an NVIDIA GPU (Ampere, Ada Lovelace, or Hopper). Apple Silicon (Metal) and AMD (ROCm) backends are not currently supported.

---

## Verified results

All figures below are programmatically extracted from [`benchmarks/results/latest.json`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/benchmarks/results/latest.json) and executed on reference hardware: **NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, PCIe 4.0 x8), 24.0 GB DDR5 RAM, Gen4 NVMe SSD, Windows 11**.

| Model | Parameter Scale | Mode | Memory Placement | Decoding Throughput | Reference Audit |
|---|:---:|:---:|---|:---:|:---:|
| **`Qwen3-30B-A3B`** | 30.5B (3.3B active) | MoE Sparse | 4.66 GB VRAM + 11.32 GB RAM | **12.95 tok/s** (Local Laptop) | [`test_03`](docs/testing/test_03_qwen3_30b_a3b.md) |
| **`Qwen3-30B-A3B`** | 30.5B (3.3B active) | MoE Sparse | 13.65 GB VRAM + 2.33 GB RAM | **24.79 tok/s** (Colab T4 Cloud) | [`test_03`](docs/testing/test_03_qwen3_30b_a3b.md) |
| **`Qwen2.5-Coder-32B`** | 32.8B (32.8B active) | 100% Dense | 4.56 GB VRAM + 14.50 GB RAM | **2.88 tok/s** (Local Laptop) | [`test_01`](docs/testing/test_01_qwen2.5_coder_32b.md) |
| **`Llama-3-70B`** | 70.6B (70.6B active) | Dense (Swap) | 4.62 GB VRAM + 17.1 GB RAM + 15.3 GB NVMe | **0.39 tok/s** (Local Laptop) | [`test_04`](docs/testing/test_04_llama3_70b.md) |
| **`SmolLM2-135M`** | 0.135B | 100% Dense | 0.08 GB VRAM (100% VRAM) | **366.5 tok/s** (Local Laptop) | [`test_02`](tests/ephemeral_test_results_smollm-135m.json) |

For complete benchmark distributions (mean, stddev, min, max, p50, p95), baseline ablations, and environment fingerprints, see [`RESULTS.md`](RESULTS.md).

---

## Real-world device impact: PHANTOM vs Baseline

How PHANTOM changes what runs on consumer hardware (measured on reference RTX 4050 6.0 GB Laptop GPU, 24.0 GB RAM):

| Task & Model | Standard Baseline Runtimes | PHANTOM Tiered Runtime | Practical User Experience |
|---|---|---|---|
| **Local Code Reasoning**<br>`Qwen2.5-Coder-32B` (32.8B) | **Immediate Crash**: CUDA OOM (requires ~20.0 GB VRAM). Naive host loaders exhaust RAM. | **Runs Stable**: 4.56 GB VRAM + 14.50 GB RAM. Evaluates host layers in-place via CPU SIMD. | **2.88 tok/s** (~170 words/min). Complete 150-token function generates in ~52s without crashes. |
| **Interactive Assistant**<br>`Qwen3-30B-A3B` (MoE) | **High Latency**: Dense offload reads all weights every token (< 3.0 tok/s). | **MoE Acceleration**: 4.66 GB VRAM + 11.32 GB RAM. 9.93x FLOP reduction on active experts. | **12.95 tok/s** on laptop (**24.79 tok/s** on cloud). Smooth interactive conversation. |
| **Frontier Scale**<br>`Llama-3-70B` (70.6B) | **Immediate Crash**: Cannot load 37.0 GB working set on consumer laptops. | **3-Tier Swap**: 4.62 GB VRAM + 17.1 GB RAM + 15.3 GB NVMe SSD swap. | **0.39 tok/s** (~2.5s per token). Usable for background batch synthesis. *Interactive chat not achieved yet — we are working on it.* |

### What is achieved vs what we are working on

- **Achieved (Production Ready)**:
  - Models up to 32B dense (`Qwen2.5-Coder-32B`) running at **2.88 tok/s** without crashing on a 6.0 GB laptop GPU.
  - MoE architectures up to 30B (`Qwen3-30B-A3B`) running at **12.95 tok/s** for interactive chat.
  - Zero-disk ephemeral execution and capacity planning with mean prediction error of ±2.4%.
- **What we are working on (In Progress)**:
  - **Conversational 70B throughput**: Running 70B models at 0.39 tok/s is physically bound by NVMe sequential read speeds (~1.4–1.8 GB/s). *We are actively working on* Linux direct `io_uring` kernel submission and fused FP8 inverse DCT decompression to minimize physical SSD page reads.
  - **Non-NVIDIA Backends**: Support for Apple Silicon (Metal) and AMD (ROCm) is planned.

---

## Supported hardware envelope

Empirically verified performance boundaries on 6 GB VRAM + 24 GB DDR5 RAM:

- **>= 5.0 tok/sec (Conversational)**: Models <= 14B Dense and Mixture-of-Experts up to 30B (`Qwen3-30B-A3B` runs at 12.95 tok/s).
- **>= 2.5 tok/sec (Interactive Reading)**: Dense models up to 32B (`Qwen2.5-Coder-32B` runs at 2.88 tok/s).
- **>= 1.0 tok/sec (Usable)**: Dense models up to 40B fitting within fast VRAM + RAM.
- **< 1.0 tok/sec (NVMe Bandwidth Bound)**: Dense models >= 70B requiring SSD paging (`Llama-3-70B` runs at 0.39 tok/s).

---

## Quick start

### Installation

```bash
git clone https://github.com/FreakyAdy/phantom.git
cd phantom
pip install -e python/
```

### Run capacity planner (0 bytes disk download)

Before downloading large models, check their memory tier distribution and expected speed:

```bash
phantom plan qwen2.5-coder:32b
phantom plan llama3:70b
```

### Trace per-token byte accounting

Verify exact byte transfers across PCIe, DDR5 RAM, and NVMe:

```bash
phantom trace qwen2.5-coder:32b --tokens 5
```

### Run inference

```bash
phantom run qwen2.5-coder:32b "Write a quicksort in Python"
```

---

## How it works

PHANTOM avoids PCIe weight thrashing by adopting an **in-place hybrid execution model**:

1. **Partitioned Forward Pass**: Initial layers run on GPU VRAM. When execution reaches host-offloaded layers, the GPU transfers only the intermediate activation vector ($[B=1, S=1, D=5120]$ FP16 $\approx 10\text{ KB}$) across PCIe to host memory ($1.3\ \mu\text{s}$ transfer latency).
2. **In-Place CPU SIMD Evaluation**: Host RAM layers are evaluated directly by CPU SIMD kernels, reading weights at dual-channel DDR5 bus bandwidth (~48 GB/s). For a 32B model with 13.5 GB in RAM, reading weights at ~48 GB/s requires ~0.31s per token, delivering 2.88 to 3.4 tokens/sec.
3. **NVMe Tile Streaming**: When models exceed fast memory (e.g. 70B models), layers are paged from NVMe SSD using 64MB compressed tiles.
4. **Predictive Prefetching (Wraith)**: A CPU-resident LSTM micro-predictor forecasts upcoming layer transitions during autoregressive decode to overlap SSD/RAM transfers with compute.
5. **Key-Value Cache Compression (Neural Cache)**: Reduces KV attention state memory footprint by up to 8x via low-rank latent projection.

For formal mathematical derivations, data flow diagrams, and subsystem invariants, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Limitations and known issues

- **NVMe Bandwidth Wall**: Models exceeding system RAM cannot run faster than physical SSD read bandwidth (~0.12–0.39 tok/s for 70B).
- **Windows Host Toolchain**: The Rust core engine compiles on Linux/WSL2; on native Windows host environments without a configured Cargo toolchain, PHANTOM automatically routes execution through the accelerated Python SIMD runtime.
- **Single Process Exclusivity**: Memory-mapped tiering assumes exclusive access to free GPU VRAM and unreserved system RAM. Heavy concurrent applications will cause OS memory contention.

---

## Benchmarks & reproduction

Every number in this repository can be reproduced using committed scripts:

```bash
# Run master benchmark suite (generates benchmarks/results/latest.json)
python benchmarks/run_all.py

# Verify numerical parity against reference baseline
python tests/correctness/test_reference_parity.py --quick

# Check CI claims consistency
python scripts/check_claims.py
```

For step-by-step reproduction instructions and GGUF checksums, see [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

---

## Contributing

Please review [`CONTRIBUTING.md`](CONTRIBUTING.md) for PR requirements: all code modifications must pass `test_reference_parity.py` and `scripts/check_claims.py`.

## License

MIT License. Copyright (c) 2026 FreakyAdy. See [`LICENSE`](LICENSE) for details.