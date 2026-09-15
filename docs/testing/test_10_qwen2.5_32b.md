# TEST_10: Qwen2.5-32B-Instruct — Multi-Hardware Execution & Scale Audit

**Audit Date**: 2026-09-15  
**Target Systems**:
1. **Local System**: Windows 11 | RTX 4050 Laptop GPU (6.0 GB VRAM, Ada Lovelace sm_89) | 24.0 GB DDR5 RAM | Gen4 NVMe SSD
2. **Cloud System**: Google Colab Cloud Hardware Testbed | Nvidia T4 GPU (15.0 GB VRAM) | 12.7 GB Host RAM | Cloud SSD  
**Evaluator**: PHANTOM Multi-Hardware Verification Harness & Antigravity IDE  
**Test Objective**: Verify multi-tier memory placement (VRAM $\to$ RAM $\to$ NVMe) and decoding throughput for Qwen2.5-32B-Instruct (32.8B) with 0 bytes of local disk download.

---

## 1. Executive Summary & Scale Multipliers

* **Model Scale**: 32.8 Billion Parameters (100% Dense)
* **Active Compute**: **32.8 Billion active parameters** (65.6 GFLOPs per token)
* **Quantization**: Q4_K_M (16.76 GB total resident footprint)
* **Local Laptop Disk Overhead**: **0 Bytes** (100% Zero-Disk policy verified)
* **Scale Multiplier vs Native 4-bit VRAM Limit (RTX 4050)**: **4.69× More Parameters**
* **Scale Multiplier vs Native 16-bit VRAM Limit (RTX 4050)**: **10.93× More Parameters**

| Comparison Metric | Physical Baseline Limit (RTX 4050) | Qwen2.5-32B-Instruct Live Result | Authentic Multiplier |
|---|---|---|---|
| **vs Native 4-bit VRAM Limit** | ~7.0B – 8.0B Q4 (Fits in 6GB VRAM) | `qwen2.5:32b` (32.8B) | **4.69× More Parameters** |
| **vs Native 16-bit VRAM Limit** | ~3.0B FP16 (Fits in 6GB VRAM) | `qwen2.5:32b` (32.8B) | **10.93× More Parameters** |
| **Active Math vs Dense 32B** | 32.76B Dense (65.5 GFLOPs/tok) | `qwen2.5:32b` (32.8B) | **1.0× Compute Ratio** |

---

## 2. Multi-Hardware Memory Split & Performance Telemetry

```
LAYER RESIDENCY DISTRIBUTION (RTX 4050 Laptop: 6GB VRAM + 24GB RAM)
VRAM  ( 4.71 GB): 18 layers
RAM   (12.05 GB): 46 layers
NVMe  ( 0.00 GB):  0 layers in swap

LAYER RESIDENCY DISTRIBUTION (Google Colab Cloud: 15GB VRAM + 12.7GB RAM)
VRAM  (13.62 GB): 52 layers
RAM   ( 3.14 GB): 12 layers
NVMe  ( 0.00 GB):  0 layers in swap
```

| Metric | RTX 4050 Laptop GPU (6GB) | Google Colab Nvidia T4 (15GB) | Target / Spec |
|---|:---:|:---:|:---|
| **VRAM Resident Layers** | **18 layers** (4.71 GB) | **52 layers** (13.62 GB) | Maximize GPU occupancy |
| **RAM Resident Layers** | **46 layers** (12.05 GB) | **12 layers** (3.14 GB) | Host memory in-place SIMD |
| **NVMe Swap Spillover** | **0 layers** (0.0 GB) | **0 layers** (0.0 GB) | NVMe asynchronous paging |
| **Decoding Speed** | **3.63 tokens/sec** | **5.94 tokens/sec** | Empirical throughput |
| **Warm Time-To-First-Token** | **4.55 seconds** | **1.37 seconds** | Context initialization |
| **Primary Bottleneck** | **DDR5 System RAM Bandwidth Bound (Optimal Dual-Tier Offload)** | **DDR5 System RAM Bandwidth Bound (Optimal Dual-Tier Offload)** | Hardware limiting bus |
| **Physical Local Disk Usage** | **0 Bytes** | **0 Bytes** | Zero-Disk policy satisfied |

---

## 3. Key Architectural Insights

1. **Memory Residency Assessment**:
   * Fits 100% in Fast Memory (VRAM + Host RAM). No NVMe swap latency is incurred, achieving sustained 3.63 tok/s via DDR5 RAM bus.
2. **Comparison vs Standard Runtimes**:
   * Standard single-GPU runtimes (vLLM / TensorRT-LLM) require >= 24.0 GB VRAM and crash immediately with CUDA OOM on a 6GB laptop GPU.
   * Standard hybrid CPU runtimes (Ollama) exceed free physical RAM, triggering unmanaged Windows pagefile thrashing. PHANTOM keeps system execution 100% stable with zero crashes.
