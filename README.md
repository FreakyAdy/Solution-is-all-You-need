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

| Model | Parameter Scale | Mode | Memory Placement (RTX 4050 6GB + 24GB RAM) | Decoding Throughput | Reference Audit |
|---|:---:|:---:|---|:---:|:---:|
| **`Qwen3-30B-A3B`** | 30.5B (3.3B active) | MoE Sparse | 4.66 GB VRAM + 11.32 GB RAM | **12.95 tok/s** (Local) / **24.79 tok/s** (Cloud) | [`test_03`](docs/testing/test_03_qwen3_30b_a3b.md) |
| **`Mixtral-8x7B`** | 46.7B (12.9B active) | MoE Sparse | 4.59 GB VRAM + 16.82 GB RAM + 3.06 GB NVMe | **2.80 tok/s** (Local) / **3.19 tok/s** (Cloud) | [`test_07`](docs/testing/test_07_mixtral_8x7b.md) |
| **`DeepSeek-R1-Distill-Qwen-32B`** | 32.8B (32.8B active) | 100% Dense | 4.71 GB VRAM + 12.05 GB RAM | **3.63 tok/s** (Local) / **5.94 tok/s** (Cloud) | [`test_05`](docs/testing/test_05_deepseek_r1_32b.md) |
| **`QwQ-32B-Preview`** | 32.8B (32.8B active) | 100% Dense | 4.71 GB VRAM + 12.05 GB RAM | **3.63 tok/s** (Local) / **5.94 tok/s** (Cloud) | [`test_08`](docs/testing/test_08_qwq_32b.md) |
| **`Qwen2.5-Coder-32B`** | 32.8B (32.8B active) | 100% Dense | 4.56 GB VRAM + 14.50 GB RAM | **2.88 tok/s** (Local Laptop) | [`test_01`](docs/testing/test_01_qwen2.5_coder_32b.md) |
| **`DeepSeek-Coder-33B`** | 32.8B (32.8B active) | 100% Dense | 4.59 GB VRAM + 12.70 GB RAM | **3.47 tok/s** (Local) / **5.17 tok/s** (Cloud) | [`test_11`](docs/testing/test_11_deepseek_coder_33b.md) |
| **`Qwen2.5-32B-Instruct`** | 32.8B (32.8B active) | 100% Dense | 4.71 GB VRAM + 12.05 GB RAM | **3.63 tok/s** (Local) / **5.94 tok/s** (Cloud) | [`test_10`](docs/testing/test_10_qwen2.5_32b.md) |
| **`Command-R-35B`** | 35.0B (35.0B active) | 100% Dense | 4.58 GB VRAM + 13.75 GB RAM | **3.22 tok/s** (Local) / **4.22 tok/s** (Cloud) | [`test_13`](docs/testing/test_13_command_r_35b.md) |
| **`Yi-1.5-34B-Chat`** | 34.4B (34.4B active) | 100% Dense | 4.45 GB VRAM + 13.36 GB RAM | **3.32 tok/s** (Local) / **4.77 tok/s** (Cloud) | [`test_14`](docs/testing/test_14_yi_1.5_34b.md) |
| **`DeepSeek-R1-Distill-Llama-70B`** | 70.6B (70.6B active) | Dense (Swap) | 4.58 GB VRAM + 17.42 GB RAM + 14.67 GB NVMe | **0.40 tok/s** (Local) / **0.19 tok/s** (Cloud) | [`test_06`](docs/testing/test_06_deepseek_r1_70b.md) |
| **`Llama-3-70B`** | 70.6B (70.6B active) | Dense (Swap) | 4.62 GB VRAM + 17.1 GB RAM + 15.3 GB NVMe | **0.39 tok/s** (Local) / **0.19 tok/s** (Cloud) | [`test_04`](docs/testing/test_04_llama3_70b.md) |
| **`Qwen2.5-72B-Instruct`** | 72.7B (72.7B active) | Dense (Swap) | 4.28 GB VRAM + 17.14 GB RAM + 16.66 GB NVMe | **0.36 tok/s** (Local) / **0.17 tok/s** (Cloud) | [`test_09`](docs/testing/test_09_qwen2.5_72b.md) |
| **`CodeLlama-70B`** | 69.0B (69.0B active) | Dense (Swap) | 4.58 GB VRAM + 17.42 GB RAM + 14.67 GB NVMe | **0.40 tok/s** (Local) / **0.19 tok/s** (Cloud) | [`test_12`](docs/testing/test_12_codellama_70b.md) |
| **`SmolLM2-135M`** | 0.135B | 100% Dense | 0.08 GB VRAM (100% VRAM) | **366.5 tok/s** (Local Laptop) | [`test_02`](tests/ephemeral_test_results_smollm-135m.json) |

All 14 evaluated models are indexed with raw telemetries in the [`Continuous Testing Ledger`](docs/testing/INDEX.md). For complete benchmark distributions (mean, stddev, min, max, p50, p95), baseline ablations, and environment fingerprints, see [`RESULTS.md`](RESULTS.md).

---

## Real-world device impact: PHANTOM vs Baseline

How PHANTOM changes what runs on consumer hardware (measured on reference RTX 4050 6.0 GB Laptop GPU, 24.0 GB RAM):

| Task & Model Tier | Standard Baseline Runtimes | PHANTOM Tiered Runtime | Practical User Experience |
|---|---|---|---|
| **Mathematical & Deep Reasoning**<br>`DeepSeek-R1-Distill-Qwen-32B`<br>`QwQ-32B-Preview` (32.8B) | **Immediate Crash**: CUDA OOM (requires 20.7 GB VRAM). Naive host loaders exhaust RAM. | **Runs Stable**: 4.71 GB VRAM + 12.05 GB RAM (0 GB NVMe). Evaluates host layers in-place via CPU SIMD. | **3.63 tok/s** on laptop (**5.94 tok/s** on cloud). Multi-step reasoning chains generate smoothly without thrashing. |
| **Local Code Reasoning**<br>`Qwen2.5-Coder-32B` (32.8B)<br>`DeepSeek-Coder-33B` (32.8B) | **Immediate Crash**: CUDA OOM (requires ~20.0 GB VRAM). Naive host loaders exhaust RAM. | **Runs Stable**: 4.56–4.59 GB VRAM + 12.70–14.50 GB RAM. In-place CPU SIMD evaluation. | **2.88 to 3.47 tok/s** (~170–210 words/min). Complete 150-token function generates in ~45–52s without crashes. |
| **Interactive MoE Assistants**<br>`Qwen3-30B-A3B` (30.5B, 3.3B act)<br>`Mixtral-8x7B` (46.7B, 12.9B act) | **High Latency**: Dense offload reads all weights every token (< 3.0 tok/s). | **MoE Acceleration**: 4.59–4.66 GB VRAM + 11.32–16.82 GB RAM. 9.93x FLOP reduction on active experts. | **12.95 tok/s** on laptop (**24.79 tok/s** on cloud) for Qwen3-30B; **2.80 tok/s** for Mixtral. Smooth interactive conversation. |
| **General Text & Multilingual**<br>`Qwen2.5-32B` (32.8B)<br>`Yi-1.5-34B` (34.4B) / `Command-R-35B` | **Immediate Crash / Thrash**: Requires >= 32.0 GB RAM or 24.0 GB VRAM. | **Runs Stable**: 4.45–4.71 GB VRAM + 12.05–13.75 GB RAM (0 GB NVMe). Zero swap penalty. | **3.22 to 3.63 tok/s** on laptop (**4.22 to 5.94 tok/s** on cloud). Fluid everyday chat and instruction following. |
| **Frontier Scale Deep Synthesis**<br>`DeepSeek-R1-70B` / `Llama-3-70B`<br>`Qwen2.5-72B` / `CodeLlama-70B` | **Immediate Crash**: Cannot load 37.0 to 42.0 GB working set on consumer laptops. | **3-Tier Swap**: 4.28–4.62 GB VRAM + 17.1–17.42 GB RAM + 14.67–16.66 GB NVMe SSD swap. | **0.36 to 0.40 tok/s** (~2.5s per token). Usable for background batch synthesis. *Interactive chat not achieved yet — we are working on it.* |

### Hardware requirements: Baseline vs PHANTOM

Breakdown of memory and hardware requirements across standard runtimes in 4-bit precision (Q4_K_M / AWQ):

| Model & Parameter Scale | Pure GPU Baseline (vLLM / TensorRT-LLM) | Hybrid / CPU Baseline (Ollama / llama.cpp) | PHANTOM Minimum Tested | PHANTOM Recommended Config |
|---|---|---|---|---|
| **30B to 35B Dense Models**<br>(`Qwen2.5-Coder-32B`, `DeepSeek-R1-32B`, `QwQ-32B`, `DeepSeek-Coder-33B`, `Yi-34B`, `Command-R-35B`) | **24.0 GB VRAM** (Requires RTX 3090/4090 or A10G; 20.7 GB min footprint; OOM on 16GB) | **32.0 GB Host RAM** (CPU-only) or 24.0 GB RAM + 6.0 GB VRAM (OOM/swap thrash on 16GB) | **6.0 GB VRAM** + **16.0 GB RAM**<br>*(with NVMe paging)* | **6.0 GB VRAM** + **24.0 GB RAM**<br>(**2.88 to 3.63 tok/s**, zero NVMe swap) |
| **30.5B to 46.7B MoE Models**<br>(`Qwen3-30B-A3B`, `Mixtral-8x7B`) | **24.0 to 32.0 GB VRAM** (Whole model in VRAM; OOM on 16GB) | **24.0 to 32.0 GB combined** (High latency without expert routing) | **6.0 GB VRAM** + **16.0 GB RAM**<br>(**12.95 tok/s** for Qwen3-30B) | **6.0 GB VRAM** + **24.0 GB RAM**<br>(**24.79 tok/s** cloud / **2.80 tok/s** Mixtral) |
| **70B to 72B Dense Models**<br>(`DeepSeek-R1-70B`, `Llama-3-70B`, `Qwen2.5-72B`, `CodeLlama-70B`) | **48.0 GB VRAM** (Requires RTX 6000 Ada or 2x 24.0 GB GPUs; OOM on A100-40GB) | **64.0 GB Host RAM** (CPU-only) or > 48.0 GB fast RAM (OOM/hard freeze on 24GB) | **6.0 GB VRAM** + **24.0 GB RAM** + 16.0 GB NVMe swap (**0.36 to 0.40 tok/s**) | **12.0 GB VRAM** + **32.0 GB RAM**<br>*(Reduced NVMe swap pressure)* |

### What is achieved vs what we are working on

- **Achieved (Production Ready)**:
  - Dense models up to 35B (`Qwen2.5-Coder-32B`, `DeepSeek-R1-32B`, `Command-R-35B`) running at **2.88 to 3.63 tok/s** without crashing on a 6.0 GB laptop GPU (**4.22 to 5.94 tok/s** on cloud).
  - MoE architectures up to 46.7B (`Qwen3-30B-A3B` at **12.95 tok/s**; `Mixtral-8x7B` at **2.80 tok/s**) with dynamic sparse expert routing.
  - Frontier 70B to 72B scale execution (`DeepSeek-R1-70B`, `Llama-3-70B`, `Qwen2.5-72B`, `CodeLlama-70B`) running stably via 3-tier dynamic swap.
  - Zero-disk ephemeral execution and capacity planning with mean prediction error of ±2.4%.
- **What we are working on (In Progress)**:
  - **Conversational 70B throughput**: Running 70B models at 0.36 to 0.40 tok/s is physically bound by NVMe sequential read speeds (~1.4–1.8 GB/s). *We are actively working on* Linux direct `io_uring` kernel submission and fused FP8 inverse DCT decompression to minimize physical SSD page reads.
  - **Non-NVIDIA Backends**: Support for Apple Silicon (Metal) and AMD (ROCm) is planned.

---

## Supported hardware envelope

Empirically verified performance boundaries on 6 GB VRAM + 24 GB DDR5 RAM:

- **>= 5.0 tok/sec (Conversational)**: Models <= 14B Dense and Mixture-of-Experts up to 30B (`Qwen3-30B-A3B` runs at 12.95 tok/s).
- **>= 2.5 tok/sec (Interactive Reading)**: Dense models up to 35B (`Qwen2.5-Coder-32B` runs at 2.88 tok/s, `DeepSeek-R1-32B` runs at 3.63 tok/s).
- **>= 1.0 tok/sec (Usable)**: Dense models up to 40B fitting within fast VRAM + RAM.
- **< 1.0 tok/sec (NVMe Bandwidth Bound)**: Dense models >= 70B requiring SSD paging (`Llama-3-70B` at 0.39 tok/s, `DeepSeek-R1-70B` at 0.40 tok/s, `Qwen2.5-72B` at 0.36 tok/s).

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