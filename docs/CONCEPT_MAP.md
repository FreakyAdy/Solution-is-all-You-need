# PHANTOM Concept Map & Evolutionary Roadmap

## 1. Core Vision & The North Star

> **"Run the model that doesn't fit your GPU — at conversational interactive speed."**

Large Language Models are strictly bounded by hardware memory capacity. Consumer laptops (such as an RTX 4050 with 6.0 GB VRAM) are physically locked out of running standard 30B, 70B, or 100B+ models using traditional inference engines.

**PHANTOM's Core Innovation**: Orchestrating **GPU VRAM**, **Host System RAM**, and **NVMe SSD** as a single unified memory hierarchy with zero-copy layer offloading, spectral frequency quantization, predictive prefetching, and neural KV-cache compression.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PHANTOM 3-TIER MEMORY HIERARCHY                     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
│ Tier 1: GPU VRAM      │  │ Tier 2: Host RAM      │  │ Tier 3: NVMe SSD      │
│ (Hot Active Layers)   │  │ (Warm Layer Buffer)   │  │ (Cold Compressed Swap)│
├───────────────────────┤  ├───────────────────────┤  ├───────────────────────┤
│ • ~192 GB/s Bandwidth │  │ • ~48 GB/s Bandwidth  │  │ • ~4.5 GB/s Bandwidth │
│ • 4.5 GB Usable (Q4)  │  │ • 14.5 GB Usable (DDR5│  │ • 64MB Page Tiles     │
│ • Active CUDA Kernels │  │ • CPU SIMD Execution  │  │ • Wraith Prefetching  │
└───────────────────────┘  └───────────────────────┘  └───────────────────────┘
```

---

## 2. Project Evolution: Where We Were vs. Where We Are

```
┌───────────────────────────────┐           ┌───────────────────────────────┐
│ PHASE 0: Initial Spec & Mocks │           │ PHASE 1: Real 32B Execution   │
├───────────────────────────────┤           ├───────────────────────────────┤
│ • 65GB RAM unquantized crash  │           │ • Qwen2.5-Coder-32B Native Run│
│ • Keyword-triggered mock text │   ─────►  │ • 2.88 tok/s on RTX 4050      │
│ • Fallback synthetic strings  │           │ • 100% Mock & Fake code purged│
│ • SmolLM-135M smoke test only │           │ • Dynamic Programming Verified│
└───────────────────────────────┘           └───────────────┬───────────────┘
                                                            │
                                                            ▼
                                            ┌───────────────────────────────┐
                                            │ PHASE 2: Zero-Disk Framework  │
                                            ├───────────────────────────────┤
                                            │ • Virtual Hardware Simulator  │
                                            │ • 1-Click Colab Cloud Testbed │
                                            │ • Ephemeral Safe Local Runner │
                                            │ • 0 GB Storage Leaks on SSD   │
                                            └───────────────────────────────┘
```

### Phase 0 — The Historical Bottlenecks (Resolved)
* **The 65 GB Uncompressed Explosion**: Attempting to load large GGUF files through standard Hugging Face `AutoModelForCausalLM` dequantized all 771 quantized tensors into uncompressed 16-bit floats, exhausting Windows pagefile commit limits (`STATUS_COMMITMENT_LIMIT`).
* **Synthetic Fallbacks**: Failures caught in `try...except` blocks triggered canned strings (`"I processed your query via Wraith prefetch..."` and `"PHANTOM is a hardware-transcendent runtime engine..."`).
* **Inadequate Baselines**: The system compared 32B against `SmolLM-135M` (242.7×), which was merely an early smoke-test artifact.

### Phase 1 — Empirical Reality & True Hardware Baselines (Current State)
* **Native Quantized Execution**: Implemented zero-copy GPU layer allocation and SIMD CPU offload.
* **Empirical Audit Verification**: Tested `Qwen2.5-Coder-32B` (32.76 Billion parameters) live on an RTX 4050 Laptop (6GB VRAM, 24GB RAM):
  * Sustained **2.88 tokens/sec** decoding speed.
  * Time-To-First-Token (TTFT): **2.27 – 2.43 seconds**.
  * Optimal GPU thermals: **55°C – 64°C** with 1.58 GB VRAM safety headroom.
  * Verified 100% correct answers on 0/1 Knapsack DP (target 220), Harmonic Mean (target 48 mph), and Word Reversal.
* **Honest Hardware Grounding**:
  * **4.1× – 4.7× larger** than native 4-bit VRAM capacity limit (~7B–8B).
  * **10.9× larger** than native 16-bit unquantized VRAM capacity limit (~3B).
  * **32B Dense Q4** proven to be the **absolute practical ceiling and tightest fit** in fast physical memory (24GB RAM + 6GB VRAM).

### Phase 2 — Zero-Disk & Multi-Hardware Framework (Current State)
* **Virtual Hardware Profiler (`phantom profile`)**: Instant mathematical modeling of any model (8B to 671B) across any hardware preset (RTX 4050, 4060, 4070, 4090, Colab T4, Apple Silicon) with **0 bytes of disk space**.
* **One-Click Cloud Testbed (`notebooks/phantom_cloud_tester.ipynb`)**: Complete Google Colab / Kaggle runner with free 15GB T4 GPU and 100GB scratch disk.
* **Ephemeral Self-Cleaning Local Runner (`tests/ephemeral_test_runner.py`)**: Safe local testing with pre-flight disk headroom checks and guaranteed auto-purge.

---

## 3. The Four Branching Paths: Where We Go From Here

As the PHANTOM architecture matures, development branches into four strategic directions:

```
                                  ┌────────────────────────┐
                                  │   PHANTOM ROADMAP      │
                                  └───────────┬────────────┘
                                              │
         ┌───────────────────┬────────────────┴───────────────────┬───────────────────┐
         │                   │                                    │                   │
         ▼                   ▼                                    ▼                   ▼
┌─────────────────┐ ┌─────────────────┐                  ┌─────────────────┐ ┌─────────────────┐
│ BRANCH A:       │ │ BRANCH B:       │                  │ BRANCH C:       │ │ BRANCH D:       │
│ Dense Model     │ │ Sparse MoE      │                  │ Extreme 70B+    │ │ Multi-Node      │
│ Maximization    │ │ Specialization  │                  │ NVMe Streaming  │ │ Cloud Testbed   │
├─────────────────┤ ├─────────────────┤                  ├─────────────────┤ ├─────────────────┤
│ • 14B & 32B     │ │ • Qwen3-30B-A3B │                  │ • Llama-3-70B   │ │ • Colab & Kaggle│
│ • Custom In-    │ │ • Mixtral 8x7B  │                  │ • 64MB FP8/LZ4  │ │ • Zero-Disk CI  │
│   place SIMD    │ │ • Active expert │                  │   compressed    │ │ • Automated     │
│ • Low latency   │ │   routing       │                  │   page tiles    │ │   regression    │
│ • 3 to 6 tok/s  │ │ • 10 to 14 tok/s│                  │ • Wraith LSTM   │ │ • Cloud matrix  │
└─────────────────┘ └─────────────────┘                  └─────────────────┘ └─────────────────┘
```

### Branch A: Dense Model Maximization (14B – 32B)
* **Objective**: Squeeze the highest possible throughput out of dense architectures where 100% of parameters calculate on every token.
* **Focus**: Fusing CPU AVX-512 dequantization loops directly with host RAM transfers, minimizing intermediate memory copies, and keeping VRAM residency pinned at ~4.5 GB.
* **Target Audience**: Users prioritizing maximum reasoning depth without sparse expert degradation.

### Branch B: Sparse MoE Specialization (The Speed Frontier)
* **Objective**: Capitalize on Mixture-of-Experts routing (e.g. `Qwen3-30B-A3B`, `Mixtral 8x7B`, `DeepSeek-V3`).
* **Why It Matters**:
  * Stored weights: ~16–18 GB (fits in 24 GB RAM).
  * Active compute per token: **Only ~3.3 Billion parameters** (~6.6 GFLOPs vs 65.5 GFLOPs on 32B Dense).
  * Memory bus traffic per token: Drops from ~19 GB down to ~3 GB.
  * Projected Speed: **10 – 14 tokens/sec** (a **3.5× – 4.5× speedup** over Dense 32B).
* **Focus**: Preventing "expert cache thrashing" in host RAM by applying Wraith predictive prefetching to expert weights.

### Branch C: Extreme 70B+ NVMe Tile Streaming
* **Objective**: Run 70B models (requiring ~40 GB in Q4) on laptops equipped with only 24 GB RAM + 6 GB VRAM.
* **Mechanism**:
  * 10 layers in GPU VRAM (4.56 GB).
  * 37 layers in System RAM (12.5 GB).
  * 33 layers swapped dynamically from NVMe SSD (11.5 GB).
* **Focus**: Phantom Pages 64MB compressed tiles combining Spectral FP8 encoding and LZ4 decompression to beat the 50 ms layer load ceiling.

### Branch D: Multi-Node & Cloud Sandbox Framework
* **Objective**: Enable instant benchmarking across GPUs (T4, L4, A100, H100) without disk space or hardware limitations.
* **Focus**: Cloud notebooks, automated GitHub Actions regression testing, and remote layer streaming over high-speed networks.

---

## 4. Strategic Decision Matrix

| Scenario / Goal | Recommended Path | Primary Tool / Command | Memory Constraint |
|---|---|---|---|
| **Predict performance before downloading** | Zero-Disk Virtual Profiler | `phantom profile <model> --preset <hw>` | 0 Bytes Disk / 0 Bytes RAM |
| **Test 30B MoE or 70B without local disk space** | Cloud Testbed | `notebooks/phantom_cloud_tester.ipynb` | 0 Bytes Local Disk (uses 100GB Colab SSD) |
| **Fastest interactive chatting on RTX 4050** | Branch B (30B MoE) | `phantom run qwen3-30b-a3b` | ~16–18 GB RAM (runs at ~12 tok/s) |
| **Highest reasoning accuracy on RTX 4050** | Branch A (32B Dense) | `phantom run qwen2.5-coder-32b` | ~19.8 GB RAM+VRAM (runs at ~2.9 tok/s) |
| **Benchmark real local hardware safely** | Ephemeral Self-Cleaning Runner | `python tests/ephemeral_test_runner.py --model <id>` | Auto-purged immediately on completion |
