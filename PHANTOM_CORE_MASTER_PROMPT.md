# ████████████████████████████████████████████████████████████████████████
# PHANTOM CORE — MASTER BUILD PROMPT
# Project: Universal Hardware-Transcendent LLM Inference Engine
# Classification: Elite Engineering Directive — Full Production Build
# Target: ANY NVIDIA GPU → Run Models Beyond Its Native Hardware Capacity
# ████████████████████████████████████████████████████████████████████████

---

## PRIME DIRECTIVE

You are a senior systems architect and ML engineer with simultaneous mastery of:
- CUDA kernel programming and GPU microarchitecture
- Quantization theory (PTQ, QAT, GPTQ, AWQ, SqueezeLLM)
- Memory hierarchy engineering (VRAM, DRAM, NVMe swap, CPU offload)
- Compiler optimization (LLVM IR, PTX, TensorRT, custom CUDA graphs)
- Operating system-level memory management and virtual address tricks
- Transformer internals (attention, KV-cache, MLP, RoPE, GQA, MLA)
- Novel research from 2020–2025 across inference efficiency

You are NOT building an MVP. You are NOT wrapping existing tools. You are building **PHANTOM CORE** — a never-before-seen inference engine that makes impossible things run. This is a real, production-grade, installable system. Every file you produce must compile and run.

---

## PROJECT NAME & IDENTITY

**PHANTOM CORE**
*"The GPU doesn't know its limits until you show it what it's missing."*

Tagline: **Run the Unreachable.**

PHANTOM CORE is a **universal** hybrid inference runtime that allows **any device** to run LLM models **far beyond its native hardware capacity** at acceptable token generation speeds. The principle is simple and universal: whatever your hardware ceiling is today, PHANTOM CORE raises it.

- A **laptop with 6GB VRAM** (RTX 4050) that normally maxes out at ~7B parameters? → PHANTOM CORE runs **70B+**.
- A **workstation with 24GB VRAM** (RTX 4090) that normally handles ~70B? → PHANTOM CORE runs **200B+**.
- A **server with 80GB VRAM** (A100) that normally tops out at ~180B? → PHANTOM CORE runs **400B+**.

This is achieved through a multi-layered stack of original innovations layered on top of established techniques. After deployment, the same hardware runs models that were previously impossible — not by magic, but by architecture. PHANTOM CORE automatically detects your hardware profile and adapts its entire optimization stack to maximize what YOUR specific device can achieve.

---

## THE INNOVATION STACK — WHAT MAKES THIS ORIGINAL

These are the PHANTOM CORE original innovations. You must implement ALL of them:

### INNOVATION 1 — "WRAITH LAYERS" (Predictive Layer Pre-Eviction)
Standard CPU offload moves GPU layers to CPU reactively. WRAITH LAYERS uses a trained **micro-predictor** (a 200-parameter LSTM) that watches attention head activation patterns across the last N tokens and **pre-fetches the next required layers from CPU/NVMe 2–3 steps ahead** before they are needed. This hides the PCIe transfer latency behind compute — critical on every tier, from laptop PCIe Gen3 to server-grade Gen5. The predictor is model-specific AND hardware-specific, auto-calibrated during a 60-second warm-up run to match YOUR device's bandwidth and latency profile. Result: near-zero layer-swap stall time regardless of hardware tier.

### INNOVATION 2 — "SPECTRAL QUANTIZATION" (Frequency-Domain Weight Compression)
Rather than quantizing weights by magnitude (INT4/INT8), SPECTRAL QUANT applies a **1D DCT (Discrete Cosine Transform)** to weight matrices row-by-row, retains only the top-K frequency coefficients (typically top 12–20% of coefficients carry 94%+ of the signal energy), stores coefficients in FP8, and reconstructs at inference time using an ultra-fast inverse DCT implemented as a custom CUDA kernel. This achieves ~85–90% memory reduction on MLP weight matrices with less perplexity loss than INT4 because it preserves the spectral structure of weight distributions. Calibrate K per-layer using Fisher Information approximation.

### INNOVATION 3 — "NEURAL CACHE" (Learned KV-Cache Compression)
The KV-cache is a VRAM killer. PHANTOM CORE trains a **per-model 3-layer autoencoder** (8M parameters, trained in 4 minutes on 50 sample prompts) that compresses KV-cache entries from dimension D to D/8 before storage, and decompresses on attention retrieval. This is different from existing work because: (a) the autoencoder is specialized per-model not universal, (b) it is calibrated to the statistical distribution of KV activations of THAT specific model on YOUR hardware, (c) decompression runs in a fused CUDA kernel alongside the attention computation. This allows 8× longer context on the same VRAM.

### INNOVATION 4 — "PHANTOM PAGES" (NVMe-Backed Virtual VRAM)
PHANTOM CORE implements a **GPU virtual memory manager** that treats NVMe SSD space as a third tier of memory (below VRAM → System RAM → NVMe). Using CUDA Unified Memory with custom page-fault hooks, layers not currently active are **serialized to NVMe in a compressed format** (LZ4-compressed BF16 tensors). A background I/O daemon manages a prefetch queue. This is not the same as llama.cpp's mmap — PHANTOM CORE manages the paging at the layer granularity AND tracks access frequency to build a **hot/cold layer LRU map** that persists between runs, getting smarter over time. The page manager auto-scales its tier sizes based on detected hardware: a laptop might use 6GB VRAM / 32GB RAM / 100GB NVMe, while a server might use 80GB VRAM / 512GB RAM / 2TB NVMe — the architecture adapts to exploit whatever resources are available.

### INNOVATION 5 — "ADAPTIVE COMPUTE ROUTING" (Sparse Activation Exploitation)
Recent research confirms that 30–70% of MLP neurons are inactive per token. PHANTOM CORE implements **runtime sparsity routing** — a learned gate (16-parameter linear probe per MLP block) that predicts which neurons will activate before the full MLP runs. Inactive neurons are skipped entirely via masked matrix multiplication using custom sparse CUDA kernels. Combined with Spectral Quantization on active weights only, this yields dramatic per-token FLOP reduction. The gates are calibrated per-model per-domain via a 2-minute profiling run.

### INNOVATION 6 — "CHRONOS SCHEDULER" (Time-Sliced Multi-Model Execution)
PHANTOM CORE allows **multiple models to coexist in the memory hierarchy simultaneously** using a time-sliced execution scheduler. Model A's active layers occupy VRAM. Model B's layers sit in a compressed staging area in system RAM. When a request arrives for Model B, the Chronos Scheduler performs a sub-400ms "model context switch" — evicting Model A's residual states to a checkpoint, loading Model B's hot layers to VRAM. This makes PHANTOM CORE a full multi-model inference server on a laptop.

### INNOVATION 7 — "RESONANCE SAMPLING" (Hardware-Aware Token Generation)
Most samplers (temperature, top-p, top-k) are GPU-agnostic. PHANTOM CORE's RESONANCE SAMPLER is aware of the GPU's current thermal state, memory bandwidth utilization, and prefetch queue depth. When the GPU is thermal-throttling or the PCIe pipe is saturated, RESONANCE SAMPLER temporarily reduces beam width and increases top-k cutoff to generate fewer but higher-quality candidates, maintaining perceptual quality while reducing compute. When the system is under-loaded, it widens beams. This yields consistent perceived quality across load conditions.

---

## FULL SYSTEM ARCHITECTURE — BUILD ALL OF THIS

```
phantom-core/
│
├── core/                          # Rust core runtime
│   ├── src/
│   │   ├── main.rs                # Entry point, CLI parser
│   │   ├── engine.rs              # Main inference orchestrator
│   │   ├── memory/
│   │   │   ├── vram_manager.rs    # VRAM allocation tracker
│   │   │   ├── phantom_pages.rs   # NVMe virtual VRAM manager (Innovation 4)
│   │   │   ├── cpu_offload.rs     # System RAM layer staging
│   │   │   └── lru_map.rs         # Persistent hot/cold layer map
│   │   ├── scheduler/
│   │   │   ├── chronos.rs         # Multi-model time-slice scheduler (Innovation 6)
│   │   │   ├── wraith_prefetch.rs # Predictive layer prefetcher (Innovation 1)
│   │   │   └── io_daemon.rs       # Background NVMe I/O thread
│   │   ├── sampler/
│   │   │   └── resonance.rs       # Hardware-aware sampler (Innovation 7)
│   │   └── ipc/
│   │       └── server.rs          # Unix socket / named pipe IPC server
│
├── kernels/                       # CUDA C++ custom kernels
│   ├── spectral_quant/
│   │   ├── dct_compress.cu        # 1D DCT weight compression (Innovation 2)
│   │   ├── idct_reconstruct.cu    # Fused iDCT reconstruct kernel
│   │   └── fisher_calibrate.cu    # Fisher Information K-selection
│   ├── neural_cache/
│   │   ├── kv_encode.cu           # KV-cache autoencoder encode kernel
│   │   ├── kv_decode.cu           # KV-cache autoencoder decode kernel (fused with attn)
│   │   └── ae_train.cu            # Fast autoencoder calibration (Innovation 3)
│   ├── sparse_moe/
│   │   ├── gate_predict.cu        # Sparsity gate inference (Innovation 5)
│   │   ├── sparse_matmul.cu       # Masked sparse matrix multiply
│   │   └── gate_calibrate.cu      # Gate calibration routine
│   ├── attention/
│   │   ├── flash_attn_v3.cu       # Custom FlashAttention-3 for Ampere/Ada
│   │   └── gqa_kernel.cu          # Grouped Query Attention fused kernel
│   └── CMakeLists.txt
│
├── python/                        # Python orchestration layer
│   ├── phantom/
│   │   ├── __init__.py
│   │   ├── loader.py              # Model weight loader (GGUF/safetensors/HF)
│   │   ├── calibrate.py           # Full calibration pipeline runner
│   │   ├── wraith_lstm.py         # Wraith LSTM micro-predictor (Innovation 1)
│   │   ├── neural_cache_ae.py     # KV autoencoder trainer (Innovation 3)
│   │   ├── spectral_analyzer.py   # Spectral K selection (Innovation 2)
│   │   ├── model_profiles/
│   │   │   ├── llama3_70b.py      # Layer map for Llama-3 70B (example profile)
│   │   │   ├── mistral_22b.py     # Layer map for Mistral 22B (example profile)
│   │   │   ├── qwen2_72b.py       # Layer map for Qwen2 72B (example profile)
│   │   │   ├── auto_detect.py     # Auto-detect model architecture (PRIMARY ENTRY POINT)
│   │   │   └── hardware_detect.py # Auto-detect GPU/RAM/NVMe and build hardware tier profile
│   │   └── api/
│   │       ├── openai_compat.py   # OpenAI-compatible REST API
│   │       └── websocket_stream.py # WebSocket token streaming
│
├── ui/                            # Electron + React control panel
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── VRAMGauge.tsx      # Live VRAM/RAM/NVMe usage
│   │   │   ├── LayerMap.tsx       # Visual layer residency map (hot/cold)
│   │   │   ├── ThermalMonitor.tsx # GPU temp, power, throttle state
│   │   │   ├── InferenceChat.tsx  # Built-in chat UI
│   │   │   ├── ModelManager.tsx   # Download, quantize, load models
│   │   │   └── CalibrationWizard.tsx # Walk through calibration steps
│   │   └── hooks/
│   │       ├── usePhantomBridge.ts # IPC bridge to Rust core
│   │       └── useMetrics.ts       # Real-time perf metrics
│
├── calibration/                   # Calibration data & scripts
│   ├── run_calibration.py         # Master calibration runner
│   ├── datasets/
│   │   └── calibration_prompts.jsonl # 500 diverse calibration prompts
│   └── profiles/                  # Saved calibration profiles per model
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
│       ├── bench_spectral_quant.py
│       ├── bench_wraith_prefetch.py
│       └── bench_full_pipeline.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── INNOVATIONS.md
│   ├── INSTALL.md
│   └── API.md
│
├── install.sh                     # One-command installer
├── Cargo.toml
├── pyproject.toml
└── README.md
```

---

## TECHNOLOGY STACK — NON-NEGOTIABLE

| Layer | Technology | Why |
|---|---|---|
| Core Runtime | **Rust** (tokio async) | Zero-cost memory safety, fearless concurrency, C FFI to CUDA |
| GPU Kernels | **CUDA C++ 12.x** | Direct PTX control, custom memory layout |
| Python Layer | **Python 3.11+** (PyTorch 2.3+) | Model loading, calibration, autoencoder training |
| API Server | **FastAPI + uvicorn** | OpenAI-compatible endpoint |
| UI | **Electron + React + TypeScript** | Cross-platform desktop control panel |
| Build System | **CMake + Cargo + uv** | Unified build pipeline |
| Compression | **LZ4 (Rust crate)** for NVMe pages, **custom FP8** for spectral coefficients |
| IPC | **Unix sockets (Linux) / Named Pipes (Windows)** between Python ↔ Rust |
| Config | **TOML** for system config, **MessagePack** for binary state |

---

## DETAILED IMPLEMENTATION INSTRUCTIONS

### PHASE 1 — FOUNDATION (Build First)

**Step 1.1 — CUDA Kernel Suite**

Build the following CUDA kernels with full test harnesses:

```cuda
// FILE: kernels/spectral_quant/dct_compress.cu
// PURPOSE: Compress weight matrix W [M x N] using row-wise 1D DCT
// INPUT: FP16/BF16 weight matrix on device
// OUTPUT: FP8 DCT coefficients + uint16 K-mask per row
// MUST: Use shared memory tiling (32x32 tiles), handle non-power-of-2 N
// MUST: Implement both Type-II DCT and its inverse in the same file
// MUST: The inverse (iDCT) kernel must be fuseable with downstream GEMM
// TARGET PERF: Process a 4096x14336 MLP weight matrix in < 8ms on RTX 4050
```

```cuda
// FILE: kernels/neural_cache/kv_decode.cu  
// PURPOSE: Decompress KV-cache entries from D/8 back to D DURING attention
// This kernel replaces the standard attention kernel entirely
// INPUT: Compressed KV entries [B x S x D/8] + Autoencoder decoder weights
// OUTPUT: Attention output [B x S x D] — computed without materializing full KV
// MUST: Fuse the decode + attention score computation + softmax into one kernel
// MUST: Use FlashAttention-style tiling to avoid materializing O(S^2) score matrix
// MUST: Handle GQA (Grouped Query Attention) natively
```

```cuda
// FILE: kernels/sparse_moe/sparse_matmul.cu
// PURPOSE: Execute MLP with inactive neurons skipped
// INPUT: Activation vector x [D], Weight W [D_ffn x D], Gate mask [D_ffn] (bool)
// OUTPUT: Result vector [D_ffn] — only computing rows where gate_mask=True
// MUST: Batch the active rows into warp-aligned groups before dispatch
// MUST: Outperform dense GEMM when sparsity > 40%
// MUST: Provide a fallback to dense GEMM when sparsity < 30% (no overhead)
```

**Step 1.2 — Rust Memory Manager**

```rust
// FILE: core/src/memory/phantom_pages.rs
// This is the NVMe virtual VRAM manager
// 
// Implement struct PhantomPageManager with:
//   - new(nvme_path: PathBuf, capacity_gb: f32) -> Self
//   - async fn evict_layer(&self, layer_id: u32, tensor: CudaTensor) -> Result<PageHandle>
//     Serializes tensor to LZ4-compressed BF16, writes to NVMe at pre-allocated offset
//   - async fn load_layer(&self, handle: PageHandle) -> Result<CudaTensor>  
//     Reads from NVMe, decompresses, uploads to VRAM via cudaMemcpyAsync
//   - fn update_lru(&self, layer_id: u32)
//     Updates access frequency map (persisted to ~/.phantom/lru_state.msgpack)
//   - async fn prefetch_hint(&self, layer_ids: Vec<u32>)
//     Queues layers for background loading without blocking caller
//
// MUST: Use io_uring on Linux for async NVMe I/O (tokio-uring crate)
// MUST: Use pre-allocated NVMe file to avoid fragmentation (phantom_swap.bin)
// MUST: Track LRU state across restarts (persisted map)
// MUST: Achieve < 50ms load time for a single 70B model layer on NVMe Gen4
```

**Step 1.3 — Wraith LSTM Predictor**

```python
# FILE: python/phantom/wraith_lstm.py
#
# Implements the micro-predictor for layer prefetching
#
# Architecture:
#   - Input: sequence of (attention_entropy, activation_L2_norm, token_position_bucket)
#     per transformer layer, over last 16 tokens — shape [16, num_layers, 3]
#   - LSTM: 2-layer, hidden_dim=64, bidirectional=False  
#   - Output: probability vector over num_layers — which layers will be needed next
#   - Training: online, using layer access logs from the current session
#     Update every 50 tokens using backprop through last 200 steps
#
# class WraithPredictor:
#   def __init__(self, num_layers: int, device: str)
#   def observe(self, layer_id: int, attn_entropy: float, l2_norm: float, tok_pos: int)
#     Record that this layer was accessed with these activation stats
#   def predict_next(self, horizon: int = 3) -> List[int]
#     Return top-k layer IDs most likely needed in the next horizon steps
#   def update_online(self)
#     Run one gradient step on the last 200 observations
#   def save(self, path: Path) / def load(cls, path: Path)
#     Persist trained predictor per model
#
# MUST: Run on CPU (not GPU) to not compete for VRAM
# MUST: predict_next() must complete in < 1ms
# MUST: Work correctly even before convergence (random good baseline)
```

---

### PHASE 2 — CALIBRATION PIPELINE

The calibration pipeline runs ONCE per model and saves a profile. It must complete in under 10 minutes.

```python
# FILE: calibration/run_calibration.py
#
# Master calibration runner. Usage:
#   python run_calibration.py --model /path/to/model --output profiles/llama70b.json
#
# Calibration pipeline (run in order):
#
# STEP A — Layer Profiling (2 min)
#   Load model with NO optimizations.
#   Run 50 diverse prompts from calibration_prompts.jsonl.
#   For each layer, record:
#     - Peak VRAM usage
#     - Activation L2 norms (mean, variance)
#     - Attention entropy distribution  
#     - MLP neuron activation frequency
#     - Forward pass time
#   Save: layer_profile.json
#
# STEP B — Spectral K Selection (1 min)
#   For each MLP weight matrix, compute 1D DCT.
#   Find minimum K (coefficients to retain) such that reconstruction error
#   causes < 0.5% increase in perplexity vs full precision.
#   Use Fisher Information as a per-layer importance weight for the K budget.
#   Save: spectral_k_map.json  {layer_id: k_value}
#
# STEP C — KV Autoencoder Training (4 min)
#   Collect KV activations from the 50 calibration runs.
#   Train a 3-layer autoencoder:
#     Encoder: Linear(D, D//4) -> GELU -> Linear(D//4, D//8)
#     Decoder: Linear(D//8, D//4) -> GELU -> Linear(D//4, D)
#   Loss: MSE reconstruction + 0.01 * L2 regularization
#   Train for 20 epochs using AdamW, lr=3e-4
#   Validate reconstruction error is < 2% on held-out KV samples
#   Export decoder weights to CUDA-compatible FP16 format for kernel loading
#   Save: kv_autoencoder.pt, kv_ae_cuda_weights.bin
#
# STEP D — Sparsity Gate Calibration (2 min)
#   For each MLP block, train a linear gate:
#     gate = sigmoid(W_gate @ x + b_gate)   where W_gate is [D_ffn, D]
#   Train to predict neuron activation binary mask (threshold 0.01 of max activation)
#   Loss: Binary cross-entropy
#   Target: > 85% precision on predicting which neurons activate
#   If F1 < 0.70 for a layer, disable sparse routing for that layer (fallback to dense)
#   Save: gate_weights.npz, gate_config.json {layer_id: {enabled: bool, threshold: float}}
#
# STEP E — Wraith Warm-up (1 min)
#   Run 30 additional prompts.
#   Record layer access sequences.
#   Do 5 initial training epochs on the Wraith LSTM.
#   Save: wraith_init.pt
#
# OUTPUT: Unified profile — profiles/{model_hash}.phantom
#   JSON manifest + binary payloads (autoencoder weights, gate weights, spectral maps)
#   This file is the "unlock key" that makes PHANTOM CORE work for this model.
```

---

### PHASE 3 — INFERENCE ENGINE (The Heart)

```rust
// FILE: core/src/engine.rs
//
// The PhantomEngine orchestrates everything.
//
// pub struct PhantomEngine {
//   model_config: ModelConfig,
//   phantom_profile: PhantomProfile,      // loaded .phantom calibration file
//   vram_manager: VRAMManager,
//   phantom_pages: PhantomPageManager,
//   cpu_offload: CpuOffloadManager,
//   chronos: ChronosScheduler,
//   wraith: WraithPrefetchController,     // Rust wrapper calling Python LSTM via PyO3
//   resonance: ResonanceSampler,
// }
//
// impl PhantomEngine {
//   pub async fn new(model_path: &Path, profile_path: &Path, config: EngineConfig) -> Result<Self>
//     Load weights in priority order: load hot layers to VRAM, warm layers to RAM, cold to NVMe
//     Apply Spectral Quantization to all MLP layers using stored K-map
//     Load KV autoencoder weights into GPU constant memory
//     Initialize all sub-systems
//
//   pub async fn generate(&self, prompt: &str, params: SamplingParams) -> TokenStream
//     Main generation loop — runs until EOS or max_tokens:
//       1. Tokenize prompt
//       2. For each transformer layer in forward pass:
//          a. Check if layer is in VRAM (VRAMManager)
//          b. If not: wait for Wraith prefetch OR trigger emergency load
//          c. Run layer with full optimization stack:
//             - Sparse routing (skip inactive neurons via gate)
//             - iDCT weight reconstruction (Spectral Quant)
//             - Fused KV-compressed attention (Neural Cache kernel)
//          d. Report access to Wraith predictor
//          e. Evict LRU layers from VRAM if needed (Phantom Pages)
//       3. Sample token via Resonance Sampler (reads GPU thermal state)
//       4. Update Wraith LSTM online (every 50 tokens)
//       5. Yield token to caller
//
//   pub fn memory_report(&self) -> MemoryReport
//     Returns {vram_used_mb, ram_used_mb, nvme_used_mb, hot_layers, cold_layers}
// }
```

---

### PHASE 4 — API SERVER

```python
# FILE: python/phantom/api/openai_compat.py
#
# OpenAI-compatible API server
# Implements: POST /v1/chat/completions (streaming + non-streaming)
#             POST /v1/completions
#             GET  /v1/models
#             GET  /v1/health
#             GET  /v1/metrics  (PHANTOM CORE extension — returns perf stats)
#
# /v1/metrics returns:
# {
#   "vram_mb": 5821,
#   "ram_mb": 22400,
#   "nvme_mb": 45000,
#   "layer_residency": {"vram": [0,1,2,...12], "ram": [13..40], "nvme": [41..79]},
#   "wraith_accuracy_pct": 87.3,
#   "kv_compression_ratio": 7.8,
#   "active_sparsity_pct": 61.2,
#   "tok_per_sec": 4.7,
#   "thermal_state": "nominal",
#   "throttle_active": false
# }
#
# All model communication goes through IPC to Rust core.
# Server must handle concurrent requests via asyncio + request queue.
# Max concurrent generations: configurable (default 1 for laptop, 3 for desktop)
```

---

### PHASE 5 — UI CONTROL PANEL

```typescript
// FILE: ui/src/components/LayerMap.tsx
//
// The centerpiece visual of the PHANTOM CORE UI.
// Shows a live heat-map grid of transformer layers.
// Each cell = one layer. Color indicates residency:
//   VRAM (hot) = bright amber
//   RAM (warm) = soft blue
//   NVMe (cold) = dark slate
//   Currently executing = pulsing green
//   Being prefetched = blinking yellow
//   
// Hovering a cell shows: layer_id, type (attn/mlp), size_mb, last_access_ms_ago
// Click a cell to force-pin it to VRAM (persists until manually unpinned)
// Updates every 200ms via WebSocket from the metrics endpoint
//
// The grid arranges layers left-to-right, top-to-bottom like a reading layout.
// Layers are NOT shown as a vertical list — they form a 2D grid (sqrt(N) x sqrt(N))
// This gives an immediate spatial intuition of which "region" of the model is hot.
```

---

### PHASE 6 — INSTALLER

```bash
#!/bin/bash
# FILE: install.sh
# ONE-COMMAND INSTALLER — must work on Ubuntu 22.04+ and Windows 11 (WSL2)
#
# Checks:
#   - NVIDIA GPU detected (nvidia-smi) — works with ANY NVIDIA GPU (Pascal+)
#   - CUDA 12.x installed
#   - Rust installed (or installs via rustup)
#   - Python 3.11+ available
#   - Available NVMe space (requires 50GB minimum, recommends more for larger models)
#   - System RAM (warns if < 16GB, recommends based on target model size)
#
# Hardware Auto-Detection (UNIVERSAL):
#   Detects GPU model, VRAM size, PCIe generation/width, RAM amount, NVMe speed.
#   Classifies device into a hardware tier and prints:
#     - Current native model ceiling (what you can run WITHOUT Phantom Core)
#     - Phantom-enhanced ceiling (what you CAN run WITH Phantom Core)
#     - Recommended model sizes for optimal tok/sec performance
#   Example output:
#     [PHANTOM CORE] Hardware Tier: LAPTOP (RTX 4050, 6GB VRAM, 32GB RAM, NVMe Gen4)
#     [PHANTOM CORE] Native ceiling: ~7B parameters
#     [PHANTOM CORE] Phantom ceiling: ~70B+ parameters @ ≥3 tok/sec
#     [PHANTOM CORE] Sweet spot: 30B–70B parameters @ 5–8 tok/sec
#
# Installs:
#   1. Build CUDA kernels (cmake --build) — auto-selects compute capability for YOUR GPU
#   2. Build Rust core (cargo build --release)
#   3. Install Python package (pip install -e .)
#   4. Install Electron UI (npm install && npm run build)
#   5. Create ~/.phantom/ directory structure
#   6. Generate hardware_profile.toml (persisted hardware tier config)
#   7. Register phantom-core as a systemd service (optional)
#   8. Add `phantom` CLI to PATH
#
# Post-install:
#   - Print ASCII banner
#   - Run hardware detection and print full compatibility report with tier classification
#   - Show what models are now reachable on this hardware
#   - Prompt user to run first calibration for their chosen model
```

---

## PERFORMANCE TARGETS — NON-NEGOTIABLE

These are the bench targets PHANTOM CORE must hit. Write benchmarks that verify each.

### Universal Targets (All Hardware Tiers)

| Metric | Target | Benchmark File |
|---|---|---|
| Wraith prefetch accuracy | ≥ 80% | bench_wraith_prefetch.py |
| Spectral Quant perplexity delta | ≤ 1.2 PPL vs FP16 | bench_spectral_quant.py |
| KV Autoencoder reconstruction error | ≤ 2% cosine distance | bench_neural_cache.py |
| Sparsity gate precision | ≥ 85% on active neurons | bench_sparse_routing.py |
| Calibration time (any model) | ≤ 10 minutes | bench_calibration.py |
| Layer load from NVMe Gen4 | ≤ 50ms per layer | bench_phantom_pages.py |
| Model context switch (Chronos) | ≤ 400ms | bench_chronos.py |

### Tiered Hardware Benchmarks (Ceiling Lift Demonstration)

These targets demonstrate the universal principle: **every tier runs models far beyond its native capacity.**

| Hardware Tier | VRAM | Native Ceiling | Phantom Ceiling | Min tok/sec | Benchmark File |
|---|---|---|---|---|---|
| Laptop (e.g. RTX 4050) | 6 GB | ~7B | **70B+** | ≥ 3 | bench_full_pipeline.py --tier laptop |
| Desktop (e.g. RTX 4090) | 24 GB | ~70B | **200B+** | ≥ 12 | bench_full_pipeline.py --tier desktop |
| Server (e.g. A100) | 80 GB | ~180B | **400B+** | ≥ 25 | bench_full_pipeline.py --tier server |

The benchmark runner auto-detects your hardware tier and selects the appropriate targets. You may also specify `--tier auto` (default) to let it classify your device automatically.

---

## CODE QUALITY REQUIREMENTS

Every file must meet:

1. **No placeholders.** No `# TODO`, no `pass`, no `...`. Every function body must be fully implemented.
2. **Error handling everywhere.** Rust: `Result<T, PhantomError>` with custom error types. Python: typed exceptions with diagnostic messages. CUDA: error checking on every API call with `CUDA_CHECK()` macro.
3. **Logging.** Structured JSON logs via `tracing` (Rust) and `structlog` (Python). Log level configurable at runtime. Every major operation logs entry/exit with timing.
4. **Tests.** Every module has unit tests. Every CUDA kernel has a correctness test comparing against CPU reference implementation.
5. **Documentation.** Every public function/struct has docstring/doc-comment explaining what it does, its inputs, outputs, and any side effects.
6. **No external network calls at inference time.** Everything runs offline after install.

---

## WHAT TO BUILD — DELIVERY ORDER

Build in this exact sequence. Each phase must compile and pass tests before moving to the next:

```
PHASE 1:  CUDA kernels (spectral_quant, neural_cache, sparse_matmul)
PHASE 2:  Rust memory manager (VRAMManager, PhantomPages, LRU)
PHASE 3:  Python calibration pipeline (run end-to-end on a small test model)
PHASE 4:  Wraith LSTM predictor
PHASE 5:  Rust inference engine (integrating all CUDA kernels)
PHASE 6:  Python API server (test with curl)
PHASE 7:  Chronos multi-model scheduler
PHASE 8:  Resonance sampler
PHASE 9:  Electron UI
PHASE 10: Installer + benchmarks + documentation
```

---

## FIRST TASK — START HERE

Begin with **PHASE 1, Kernel 1**: `kernels/spectral_quant/dct_compress.cu`

Before writing any code:
1. State the mathematical definition of the 1D Type-II DCT you will implement
2. Describe your CUDA tiling strategy for the target matrix size (4096 x 14336)
3. Describe how you will handle rows where N is not a power of 2
4. Describe the FP8 encoding scheme you will use for the coefficient storage
5. State your correctness test plan (what is your reference implementation?)

Then write the complete, compilable, correct CUDA file.

After that, proceed to the next file in sequence. Do not stop. Do not summarize. Build the thing.

---

## PHILOSOPHICAL CONTRACT

> PHANTOM CORE does not ask the hardware what it can do.  
> It tells the hardware what it will do.  
> Your GPU has X GB of VRAM. Your system has Y GB of RAM. Your NVMe has Z TB of space.  
> PHANTOM CORE uses ALL of it — every byte of VRAM, every page of RAM, every block of NVMe,  
> AND the PCIe bus AND the thermal budget AND the idle CPU cycles  
> AND every microsecond of latency that can be hidden behind prefetch.  
> The model that was impossible on your hardware yesterday? It runs today.  
> That is the only acceptable outcome.  
> This is not a tool for one GPU. This is a universal law: **no hardware will ever be the bottleneck again.**

---

## EXTENDED INNOVATION RESEARCH NOTES
*(Read these for inspiration — incorporate the ideas into the code)*

**On Spectral Quantization:**
Weight matrices in transformer MLP layers exhibit strong low-frequency dominance when viewed through the DCT lens — this is NOT well-known. The intuition is that the "semantic" information in a weight matrix is encoded in the smooth, slowly-varying components (low DCT frequencies), while high-frequency components correspond to noise and fine-grained memorized patterns. By throwing away high-frequency DCT coefficients, you preserve the semantic computation while discarding memorized tokens. This actually IMPROVES generalization slightly while reducing memory. Implement per-row DCT (not 2D DCT) because rows of weight matrices correspond to individual output neurons which have independent spectra.

**On Neural Cache:**
The key insight: KV-cache entries for nearby tokens are highly correlated — their autoencoder latent representations cluster in a low-dimensional subspace. The per-model autoencoder learns this subspace. During attention, you don't need to decompress to full D — you can compute approximate attention scores directly in the latent space using a learned metric (train a small bilinear form Q' W_metric K_latent). This is even more aggressive than D/8 — it's D/8 computation for attention as well. Explore implementing this as an extension.

**On Wraith Layers:**
The biological analogy: the prefrontal cortex pre-activates expected sensory representations before the sensory input arrives (predictive coding). Wraith does the same — it pre-loads the GPU's "working memory" with what the computation will need before the computation asks. The LSTM predictor should be viewed as an "attention to future computation" module. Its latent state is a compressed representation of "computational intent."

**On Phantom Pages:**
The NVMe tier should be organized into 512MB "phantom blocks" — each block contains a fixed number of transformer layers in compressed form, along with metadata (layer IDs, compression ratio, last access timestamp). This allows the I/O daemon to issue 512MB sequential reads rather than scattered small reads, exploiting NVMe's sequential bandwidth (which on Gen4 exceeds 7GB/s on most laptop SSDs).

---

*PHANTOM CORE — Run the Unreachable.*
*Build it. Ship it. Let the world catch up.*

# ████████████████████████████████████████████████████████████████████████
# END OF MASTER PROMPT
# ████████████████████████████████████████████████████████████████████████
