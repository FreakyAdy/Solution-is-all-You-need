# PHANTOM Architecture Specification

## Overview

PHANTOM is a hardware-transcendent LLM inference engine and runtime platform designed to run models up to 10× larger than native GPU VRAM capacity.

```
┌─────────────────────────────────────────────────────────────────┐
│                        PHANTOM CLI                              │
│              phantom pull / run / list / serve / plan           │
├─────────────────────────────────────────────────────────────────┤
│                      PHANTOM UI (Web)                           │
│            Dashboard · Models · Chat · Metrics · Layers         │
├─────────────────────────────────────────────────────────────────┤
│                    PHANTOM RUNTIME                              │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐ │
│  │  Model Mgr   │  │  Phantomfile  │  │    Plugin System     │ │
│  │ pull/index   │  │   System      │  │  RAG · Tools · Cache │ │
│  └──────────────┘  └───────────────┘  └──────────────────────┘ │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐ │
│  │  GGUF Loader │  │ Format Conv.  │  │    API Gateway       │ │
│  │  + Dequant   │  │  Pipeline     │  │  Auth · RL · Ollama  │ │
│  └──────────────┘  └───────────────┘  └──────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                  PHANTOM CORE (Inference Engine)                │
│   Wraith Layers  │  Spectral Quant  │  Neural Cache            │
│   Phantom Pages  │  Adaptive Compute│  Chronos  │  Resonance   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Subsystems

### 1. Engine Layer (Rust + CUDA)
- **VRAM Manager**: Hot layer pinning, memory pressure tracking.
- **Phantom Pages**: Asynchronous direct NVMe I/O via io_uring / async thread pools.
- **CPU Offload Manager**: Dual-buffer streaming for intermediate tier.
- **LRU Map**: Dynamic resonance tracking for optimal layer placement.
- **Resonance Sampler**: Thermal-adaptive sampling penalties.
- **Chronos Scheduler**: Multi-model time-slicing and sub-400ms context switches.

### 2. Runtime Platform (Python)
- **GGUF Native Loader**: Pure PyTorch/NumPy SIMD dequantizer (Q4_K_M, Q5_K_M, Q8_0).
- **Format Converter**: Generates `.phantomw` binary format with per-layer FP8 DCT coefficients.
- **Model Registry**: Hugging Face Hub + community index resolver.
- **Phantomfile Engine**: Declarative persona and sampling configuration.
- **Hardened Gateway**: Bearer token auth, rate-limiting, and drop-in Ollama endpoints.
- **Plugin System**: Pre-request, pre-generate, on-token, and post-generate interceptors.
