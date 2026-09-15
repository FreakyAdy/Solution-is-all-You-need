# Reproducing PHANTOM Benchmarks & Results

This guide provides exact step-by-step instructions to reproduce every verified metric in [`RESULTS.md`](../RESULTS.md) on comparable consumer hardware.

---

## 1. Reference Hardware Environment

The canonical figures in `RESULTS.md` were measured on the reference testbed:
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB GDDR6, Ada Lovelace `sm_89`, PCIe 4.0 x8)
- **Host CPU**: 13th Gen Intel Core i7 / i9 (Intel64 Family 6 Model 183)
- **Host RAM**: 24.0 GB dual-channel DDR5-4800/5200 (~48.0 GB/s practical bandwidth)
- **Primary Storage**: Gen4 NVMe SSD (~1.43 GB/s sustained, ~1.95 GB/s burst read)
- **Operating System**: Windows 11 / WSL2 (Ubuntu 22.04 LTS)
- **Software Stack**: Python 3.10+, CUDA 12.6, PyTorch 2.10.0+cu126

---

## 2. Environment Setup

Clone repository and install the development package:

```bash
git clone https://github.com/FreakyAdy/phantom.git
cd phantom
pip install -e python/
```

Verify GPU visibility and driver configuration:

```bash
phantom doctor
```

Output should report PCIe generation, link width, physical VRAM, and RAM capacities.

---

## 3. Reference Models & Weights

To eliminate local disk clutter, PHANTOM supports both local GGUF models and zero-disk ephemeral execution:

| Model ID | Parameter Scale | Quantization | Source Repository / Model Tag | SHA-256 Checksum (First 16 chars) |
|---|:---:|:---:|---|---|
| `SmolLM2-135M` | 0.135B | FP32 / Q4_K_M | `HuggingFaceTB/SmolLM2-135M` | `a3b890f1d48c211e...` |
| `Qwen3-30B-A3B` | 30.5B (3.3B active) | Q4_K_M | `Qwen/Qwen3-30B-A3B-GGUF` | `f87920194bc0281a...` |
| `Qwen2.5-Coder-32B` | 32.76B | Q4_K_M | `Qwen/Qwen2.5-Coder-32B-Instruct-GGUF` | `e2a481c019d5f782...` |
| `Llama-3-70B` | 70.6B | Q4_K_M | `meta-llama/Meta-Llama-3-70B-Instruct-GGUF` | `b94c029148d21b44...` |

---

## 4. Benchmark Execution Commands

### Step 1: Verify Numerical Reference Parity
Run the numerical parity test comparing PHANTOM output against HuggingFace CPU FP32 reference logits across all 6 architectural ablation rows:

```bash
python tests/correctness/test_reference_parity.py --quick
```

**Expected Result**:
- `baseline` top-1 token agreement: **100.0%**
- Mean KL divergence on baseline: **0.0000**
- All 6 configurations report `[PASS — NUMERICALLY SOUND]`.

### Step 2: Verify Per-Token PCIe & RAM Byte Accounting
Verify that the 32B run transfers only intermediate activations across PCIe, resolving the bandwidth paradox:

```bash
phantom trace qwen2.5-coder:32b --tokens 5
```

**Expected Result**:
- `bytes_h2d_per_token`: ~10 KB (activations only)
- `implied_pcie_bandwidth_gbs`: ~$0.000033\text{ GB/s} \ll 12.8\text{ GB/s}$
- Host RAM resident weights (~12.60 GB) evaluated in-place via CPU SIMD at DDR5 memory bandwidth (~44–48 GB/s).

### Step 3: Run Full Subsystem Benchmark Suite ($N \ge 10$ Runs)
Execute the complete empirical benchmark suite covering Wraith prefetch, Spectral Quantization, Neural Cache, Phantom Pages, Chronos Scheduler, and Capacity Planner validation:

```bash
python benchmarks/run_all.py
```

This runs $N=10$ iterations per benchmark after warmup, measures variance (mean, stddev, min, max, p50, p95), tests baseline ablations, and writes the canonical results artifact:
- `benchmarks/results/latest.json`
- `benchmarks/results/history/<timestamp>.json`

### Step 4: Verify Consistency & Generated Ledgers
Verify that `RESULTS.md` matches `latest.json` without uncommitted drift:

```bash
python scripts/generate_results.py --check
```

Verify that all documentation claims conform to approved ground truth:

```bash
python scripts/check_claims.py
```

**Expected Result**:
`[PASS] 100% of numeric claims across all documents are verified & consistent.`
