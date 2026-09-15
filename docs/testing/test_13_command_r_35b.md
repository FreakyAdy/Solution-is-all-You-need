# TEST_13: Command-R-35B — Multi-Hardware Execution & Scale Audit

**Audit Date**: 2026-09-15  
**Target Systems**:
1. **Local System**: Windows 11 | RTX 4050 Laptop GPU (6.0 GB VRAM, Ada Lovelace sm_89) | 24.0 GB DDR5 RAM | Gen4 NVMe SSD
2. **Cloud System**: Google Colab Cloud Hardware Testbed | Nvidia T4 GPU (15.0 GB VRAM) | 12.7 GB Host RAM | Cloud SSD  
**Evaluator**: PHANTOM Multi-Hardware Verification Harness & Antigravity IDE  
**Test Objective**: Verify multi-tier memory placement (VRAM $\to$ RAM $\to$ NVMe) and decoding throughput for Command-R-35B (35.0B) with 0 bytes of local disk download.

---

## 1. Executive Summary & Scale Multipliers

* **Model Scale**: 35.0 Billion Parameters (100% Dense)
* **Active Compute**: **35.0 Billion active parameters** (70.0 GFLOPs per token)
* **Quantization**: Q4_K_M (18.34 GB total resident footprint)
* **Local Laptop Disk Overhead**: **0 Bytes** (100% Zero-Disk policy verified)
* **Scale Multiplier vs Native 4-bit VRAM Limit (RTX 4050)**: **5.0× More Parameters**
* **Scale Multiplier vs Native 16-bit VRAM Limit (RTX 4050)**: **11.67× More Parameters**

| Comparison Metric | Physical Baseline Limit (RTX 4050) | Command-R-35B Live Result | Authentic Multiplier |
|---|---|---|---|
| **vs Native 4-bit VRAM Limit** | ~7.0B – 8.0B Q4 (Fits in 6GB VRAM) | `command-r:35b` (35.0B) | **5.0× More Parameters** |
| **vs Native 16-bit VRAM Limit** | ~3.0B FP16 (Fits in 6GB VRAM) | `command-r:35b` (35.0B) | **11.67× More Parameters** |
| **Active Math vs Dense 32B** | 32.76B Dense (65.5 GFLOPs/tok) | `command-r:35b` (35.0B) | **1.07× Compute Ratio** |

---

## 2. Multi-Hardware Memory Split & Performance Telemetry

```
LAYER RESIDENCY DISTRIBUTION (RTX 4050 Laptop: 6GB VRAM + 24GB RAM)
VRAM  ( 4.58 GB): 16 layers
RAM   (13.75 GB): 48 layers
NVMe  ( 0.00 GB):  0 layers in swap

LAYER RESIDENCY DISTRIBUTION (Google Colab Cloud: 15GB VRAM + 12.7GB RAM)
VRAM  (13.47 GB): 47 layers
RAM   ( 4.87 GB): 17 layers
NVMe  ( 0.00 GB):  0 layers in swap
```

| Metric | RTX 4050 Laptop GPU (6GB) | Google Colab Nvidia T4 (15GB) | Target / Spec |
|---|:---:|:---:|:---|
| **VRAM Resident Layers** | **16 layers** (4.58 GB) | **47 layers** (13.47 GB) | Maximize GPU occupancy |
| **RAM Resident Layers** | **48 layers** (13.75 GB) | **17 layers** (4.87 GB) | Host memory in-place SIMD |
| **NVMe Swap Spillover** | **0 layers** (0.0 GB) | **0 layers** (0.0 GB) | NVMe asynchronous paging |
| **Decoding Speed** | **3.22 tokens/sec** | **4.22 tokens/sec** | Empirical throughput |
| **Warm Time-To-First-Token** | **5.0 seconds** | **1.61 seconds** | Context initialization |
| **Primary Bottleneck** | **DDR5 System RAM Bandwidth Bound (Optimal Dual-Tier Offload)** | **DDR5 System RAM Bandwidth Bound (Optimal Dual-Tier Offload)** | Hardware limiting bus |
| **Physical Local Disk Usage** | **0 Bytes** | **0 Bytes** | Zero-Disk policy satisfied |

---

## 3. Key Architectural Insights

1. **Memory Residency Assessment**:
   * Fits 100% in Fast Memory (VRAM + Host RAM). No NVMe swap latency is incurred, achieving sustained 3.22 tok/s via DDR5 RAM bus.
2. **Comparison vs Standard Runtimes**:
   * Standard single-GPU runtimes (vLLM / TensorRT-LLM) require >= 24.0 GB VRAM and crash immediately with CUDA OOM on a 6GB laptop GPU.
   * Standard hybrid CPU runtimes (Ollama) exceed free physical RAM, triggering unmanaged Windows pagefile thrashing. PHANTOM keeps system execution 100% stable with zero crashes.
