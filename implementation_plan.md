# PHANTOM — Complete Master Implementation Plan
### Hardware-Transcendent LLM Inference Engine & Runtime Platform
> *"Ollama runs the model that fits your GPU. PHANTOM runs the model that doesn't."*

---

## Executive Summary & Architecture Overview

PHANTOM consists of two tightly coupled layers:
1. **PHANTOM CORE** (Engine Layer): Universal hardware-transcendent inference engine powered by 7 original innovations:
   - **Wraith Layers**: 2-layer online LSTM micro-predictor predicting future layer access with ≥80% accuracy in <1ms.
   - **Spectral Quantization**: Discrete Cosine Transform (DCT) frequency compression of MLP weights into FP8 (≤1.2 PPL loss).
   - **Neural Cache**: Autoencoder-compressed KV cache achieving 8× memory reduction with ≤2% cosine reconstruction error.
   - **Phantom Pages**: 3-tier memory hierarchy (VRAM → RAM → NVMe Gen4) with direct asynchronous NVMe paging (≤50ms layer swap).
   - **Adaptive Compute Routing**: Dynamic gate predicting active neurons per token to skip inactive MLP pathways.
   - **Chronos Scheduler**: Multi-model time-slicing with sub-400ms context switches and compressed RAM staging.
   - **Resonance Sampler**: Thermal- and hardware-aware sampling dynamically adapting compute and penalties to GPU status.

2. **PHANTOM RUNTIME PLATFORM** (Ecosystem & Platform Layer):
   - **GGUF Native Loader & Dequantizer**: Zero-dependency pure PyTorch/NumPy SIMD dequantizer supporting all major quants (Q4_K_M, Q5_K_M, Q8_0, etc.) at ≥2 GB/s.
   - **Model Format Conversion Pipeline**: `.phantomw` binary format with per-layer DCT compression and automated calibration bundling.
   - **Model Lifecycle Manager**: `pull`, `list`, `show`, `rm`, `search` with Hugging Face Hub and community registry mirrors.
   - **Phantomfile System**: Declarative model persona and configuration engine with full Ollama Modelfile backwards compatibility.
   - **Unified CLI & Interactive REPL**: `phantom plan` (terminal ceiling lift calculation), live `/layers` ASCII memory residency map, `/stats`, `/doctor`, etc.
   - **Hardened API Gateway**: Drop-in Ollama compatibility (`/api/generate`, `/api/chat`, `/api/tags`, etc.), OpenAI endpoints, Bearer auth, token-bucket rate limiting, structured audit logging, and 200ms WebSocket telemetry.
   - **Plugin System**: Extensible middleware pipeline (`pre_request`, `pre_generate`, `on_token`, `post_generate`) with built-in RAG, Tool Router, and 8× compressed Prefix Context Cache.
   - **Web Dashboard**: Self-hosted dark glass-morphism web UI featuring the **Ceiling Lift Hero** and the live **LayerMap Heatmap**.

---

## Current Status Audit

### ✅ What Has Been Built So Far:
- **CUDA Kernels** (`phantom-core/kernels/`): Full suite of 11 kernels (DCT compress/reconstruct, Fisher calibration, KV autoencoder train/encode/decode, Sparse MoE gate predict/calibrate/matmul, FlashAttention-v3, GQA).
- **Python Core Innovations** (`phantom-core/python/phantom/`):
  - `wraith_lstm.py` (Innovation 1: Wraith LSTM micro-predictor)
  - `neural_cache_ae.py` (Innovation 3: KV cache autoencoder)
  - `spectral_analyzer.py` (Innovation 2: DCT spectral analyzer & Fisher K selection)
  - `loader.py` (Multi-format weight loader)
  - `calibrate.py` (5-step automated calibration pipeline)
  - `model_profiles/` (`auto_detect.py`, `hardware_detect.py`, `llama3_70b.py`, `mistral_22b.py`, `qwen2_72b.py`)
  - `api/` (`openai_compat.py`, `websocket_stream.py`)
- **Calibration Subsystem** (`phantom-core/calibration/`):
  - `run_calibration.py` CLI runner
  - `gen_prompts.py` dataset generator (451+ prompts in `calibration_prompts.jsonl`)
- **Rust Core Memory & Engine Skeletons** (`phantom-core/core/`):
  - `memory/` (`vram_manager.rs`, `phantom_pages.rs`, `cpu_offload.rs`, `lru_map.rs`)
  - `lib.rs`, `Cargo.toml`, basic `main.rs`

---

## Proposed Changes: Step-by-Step Delivery Roadmap

We will deliver the remaining components following the exact sequential order specified in both Master Prompts:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: GGUF Native Loader & SIMD Dequantizer                                        │
│ PHASE 2: PHANTOM Native Format (.phantomw) & Binary Spec                              │
│ PHASE 3: Conversion Pipeline (phantom convert)                                         │
│ PHASE 4: Model Lifecycle Manager (HuggingFace Hub & Community Registry)                │
│ PHASE 5: Phantomfile Parser, Validator & Modelfile Importer                           │
│ PHASE 6: Rust Core Engine Finalization (IPC Server, Chronos, Resonance, Full Generate)│
│ PHASE 7: Unified CLI & Interactive REPL (phantom plan, doctor, status, /layers REPL)  │
│ PHASE 8: Hardened API Gateway & Ollama Drop-in Compatibility                           │
│ PHASE 9: Plugin System (Base Protocol, RAG Connector, Tool Router, Context Cache)      │
│ PHASE 10: Web Dashboard (React SPA, Ceiling Lift, LayerMap Heatmap, Telemetry)        │
│ PHASE 11: Benchmark Suite & Full Audit Suite                                           │
│ PHASE 12: Documentation, Packaging & Installers (install.sh, install.ps1, Docs)       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 1: GGUF Native Loader & SIMD Dequantizer
**Target Files:**
- `phantom-core/python/phantom/loader/gguf_loader.py`
- `phantom-core/python/phantom/loader/format_detect.py`
- `phantom-core/python/phantom/loader/__init__.py`

**Key Specifications:**
1. Zero external binary dependency (no `llama.cpp` ctypes).
2. Pure PyTorch + NumPy memory-mapped (`mmap`) parsing of GGUF headers and tensor directories.
3. High-throughput bit-unpacking and dequantization supporting:
   - `Q4_0`, `Q4_1`, `Q4_K_S`, `Q4_K_M`
   - `Q5_K_S`, `Q5_K_M`, `Q6_K`
   - `Q8_0`, `Q8_1`, `F16`, `BF16`, `F32`
4. Dequantization correctness asserted: `max(abs(phantom - reference)) < 1e-4`.
5. Target throughput: ≥ 2 GB/s on CPU with vectorized NumPy/Torch SIMD.
6. Architecture auto-detection: LLaMA (1/2/3/3.1/3.2/3.3), Mistral / Mixtral MoE, Gemma 1/2, Qwen 1.5/2/2.5, Phi 2/3/3.5, DeepSeek (V2/V3 MLA attention), Falcon.

---

### Phase 2: PHANTOM Native Format (`.phantomw`)
**Target Files:**
- `phantom-core/python/phantom/converter/format_spec.py`
- `phantom-core/python/phantom/converter/__init__.py`

**Key Specifications:**
1. Binary header specification:
   - `[4 bytes]` Magic: `PHTW` (0x50 0x48 0x54 0x57)
   - `[4 bytes]` Version: 1
   - `[4 bytes]` Layer ID
   - `[4 bytes]` Num tensors
2. Per-tensor records:
   - `[64 bytes]` Null-padded UTF-8 tensor name
   - `[4 bytes]` Rows (uint32)
   - `[4 bytes]` Cols (uint32)
   - `[4 bytes]` `k_coefficients_per_row` (0 = BF16 raw, >0 = FP8 DCT coefficients)
   - `[4 bytes]` `compressed_bytes` (uint32)
   - `[N bytes]` Data payload (FP8 DCT coefficients or BF16)
3. Zero-copy read/write serialization routines with CRC32 integrity validation.

---

### Phase 3: Model Format Conversion Pipeline
**Target Files:**
- `phantom-core/python/phantom/converter/phantom_convert.py`

**Key Specifications:**
1. End-to-end conversion from raw GGUF to `~/.phantom/models/<model-id>/`:
   - `manifest.json`: Architecture, param count, GGUF hash, quantization metadata.
   - `config.toml`: Engine parameters for PHANTOM CORE.
   - `tokenizer/`: `tokenizer.json` and special tokens extracted from GGUF.
   - `weights/`: `embed.bf16.bin`, `layer_NNN.phantomw`, `lm_head.bf16.bin`.
   - `profile/`: Automated calibration profile (`calibration.phantom`), pre-warmed Wraith LSTM (`wraith_init.pt`), hardware profile (`hardware_profile.toml`).
2. Streaming layer-by-layer processing keeping peak RAM ≤ 24 GB on 70B models.
3. Round-trip cosine similarity target: ≥ 0.995 across all MLP layers.

---

### Phase 4: Model Lifecycle Manager
**Target Files:**
- `phantom-core/python/phantom/registry/model_manager.py`
- `phantom-core/python/phantom/registry/hf_client.py`
- `phantom-core/python/phantom/registry/downloader.py`
- `phantom-core/python/phantom/registry/index_client.py`
- `phantom-core/python/phantom/registry/__init__.py`
- `phantom-core/phantom-models/index.json`

**Key Specifications:**
1. Resolves model references:
   - `llama3:70b`, `mistral:22b`, `phi3:3.8b` -> Community Index.
   - `bartowski/Meta-Llama-3-70B-Instruct-GGUF` -> HuggingFace Hub API.
   - Local `.gguf` file path -> Conversion pipeline.
2. Resumable chunked HTTP downloader with SHA-256 verification and terminal progress bars.
3. `pull`, `list`, `show`, `rm`, `search` methods with both human table and JSON outputs.

---

### Phase 5: Phantomfile System
**Target Files:**
- `phantom-core/python/phantom/phantomfile/parser.py`
- `phantom-core/python/phantom/phantomfile/validator.py`
- `phantom-core/python/phantom/phantomfile/__init__.py`

**Key Specifications:**
1. Parser for `FROM`, `SYSTEM` (including triple-quoted multi-line), `TEMPLATE`, `PARAMETER` (temperature, top_p, top_k, repeat_penalty, seed, num_predict, context_window, stop), `PHANTOM_PARAM` (sparsity_routing, kv_compression, spectral_quant, safe_mode, tier_preference, max_vram_mb), `PLUGIN`, `MESSAGE`, `LICENSE`.
2. Seamless compatibility: Ingests any valid Ollama `Modelfile` directly.
3. Strict validator with helpful error diagnostics and model card generation (`to_modelcard()`).

---

### Phase 6: Rust Core Engine Finalization
**Target Files:**
- `phantom-core/core/src/ipc/server.rs`
- `phantom-core/core/src/ipc/mod.rs`
- `phantom-core/core/src/engine/mod.rs`
- `phantom-core/core/src/sampler/mod.rs`
- `phantom-core/core/src/scheduler/mod.rs`
- `phantom-core/Cargo.toml`

**Key Specifications:**
1. **Cross-Platform IPC Server**: Windows Named Pipes (`\\.\pipe\phantom_ipc`) and Unix Domain Sockets (`/tmp/phantom.sock`) with MessagePack framing.
2. **Full Engine Loop**: Real layer loading, VRAM residency check, Wraith-predicted prefetch trigger via async Phantom Pages, CUDA kernel dispatch, dynamic LRU eviction, and Resonance sampling.
3. **Resonance Sampler**: Thermal-state monitoring (nominal, elevated, critical via NVML / simulated fallbacks), dynamic temperature modulation, repetition penalties, and beam adjustments.
4. **Chronos Scheduler**: Multi-model time-slicing with compressed RAM staging, sub-400ms model context switching, and KV checkpoint/restore.

---

### Phase 7: Unified CLI & Interactive REPL
**Target Files:**
- `phantom-core/python/phantom/phantom_cli.py`
- `phantom-core/core/src/main.rs`

**Key Specifications:**
1. CLI commands: `pull`, `run`, `list`, `show`, `rm`, `search`, `create`, `serve`, `calibrate`, `status`, `plan`, `convert`, `doctor`, `update`.
2. **`phantom plan <model>`**: Instant resource calculation showing layer distribution across VRAM/RAM/NVMe, estimated tok/sec, and hardware ceiling lift WITHOUT downloading or loading the model.
3. **Interactive REPL**: Token streaming, `/layers` ASCII residency map, `/stats`, `/system <prompt>`, `/save <file>`, `/load <file>`, `/clear`.
4. **`phantom doctor`**: CUDA kernel verification, NVMe sequential/random read speed test, VRAM validation, and health checks.

---

### Phase 8: Hardened API Gateway & Ollama Compatibility
**Target Files:**
- `phantom-core/python/phantom/api/gateway.py`
- `phantom-core/python/phantom/api/ollama_compat.py`
- `phantom-core/python/phantom/api/__init__.py`

**Key Specifications:**
1. Hardened gateway: Bearer token authentication, per-IP token bucket rate limiting (429 with `Retry-After`), structured JSONL request logging to `~/.phantom/logs/requests.jsonl`, async FIFO request queue, CORS support.
2. Ollama Drop-in Endpoints:
   - `POST /api/generate` -> `/v1/completions`
   - `POST /api/chat` -> `/v1/chat/completions`
   - `GET /api/tags` -> `/v1/models`
   - `POST /api/pull` -> `phantom pull`
   - `DELETE /api/delete` -> `phantom rm`
   - `POST /api/show` -> `/phantom/models/<id>/profile`
   - `GET /api/ps` -> Running models status
3. Native PHANTOM Endpoints:
   - `/phantom/models/<id>/layers`, `/phantom/models/<id>/pin-layer`
   - `/phantom/hardware`, `/phantom/calibrate/<id>`
   - `WS /phantom/metrics/stream` (200ms telemetry stream)

---

### Phase 9: Extensible Plugin System
**Target Files:**
- `phantom-core/python/phantom/plugins/base.py`
- `phantom-core/python/phantom/plugins/rag_connector/plugin.py`
- `phantom-core/python/phantom/plugins/tool_router/plugin.py`
- `phantom-core/python/phantom/plugins/context_cache/plugin.py`
- `phantom-core/python/phantom/plugins/__init__.py`

**Key Specifications:**
1. Async plugin protocol intercepting `pre_request`, `pre_generate`, `on_token`, `post_generate`.
2. **RAGPlugin**: Local vector search and context injection into prompts.
3. **ToolRouterPlugin**: Function calling with `@phantom_tool` decorators and MCP protocol support.
4. **ContextCachePlugin**: Prefix KV caching storing compressed states via Neural Cache for 8× VRAM savings and sub-100ms TTFT on repeat prompts.

---

### Phase 10: Web Dashboard
**Target Files:**
- `phantom-core/ui/web/package.json`
- `phantom-core/ui/web/vite.config.ts`
- `phantom-core/ui/web/src/App.tsx`
- `phantom-core/ui/web/src/pages/Dashboard.tsx`
- `phantom-core/ui/web/src/pages/Models.tsx`
- `phantom-core/ui/web/src/pages/Chat.tsx`
- `phantom-core/ui/web/src/pages/Metrics.tsx`
- `phantom-core/ui/web/src/pages/Layers.tsx`
- `phantom-core/ui/web/src/pages/Plugins.tsx`
- `phantom-core/ui/web/src/components/CeilingLift.tsx`
- `phantom-core/ui/web/src/components/LayerMap.tsx`
- `phantom-core/ui/web/src/components/VRAMGauge.tsx`
- `phantom-core/ui/web/src/components/ThermalMonitor.tsx`

**Key Specifications:**
1. Self-hosted React SPA served by the API Gateway on `http://localhost:11411/ui`.
2. Premium dark glass-morphism aesthetic (Inter font, smooth gradients, real-time micro-animations).
3. **CeilingLift.tsx**: Hero visual showing "Without PHANTOM (7B max)" vs "With PHANTOM (70B+ capable)".
4. **LayerMap.tsx**: 2D heatmap grid showing layer residency (amber=VRAM, blue=RAM, slate=NVMe, pulsing green=active, blinking yellow=prefetching) with click-to-pin.

---

### Phase 11: Benchmark & Audit Suite
**Target Files:**
- `phantom-core/tests/benchmarks/bench_spectral_quant.py`
- `phantom-core/tests/benchmarks/bench_wraith_prefetch.py`
- `phantom-core/tests/benchmarks/bench_neural_cache.py`
- `phantom-core/tests/benchmarks/bench_sparse_routing.py`
- `phantom-core/tests/benchmarks/bench_phantom_pages.py`
- `phantom-core/tests/benchmarks/bench_chronos.py`
- `phantom-core/tests/benchmarks/bench_full_pipeline.py`
- `phantom-core/tests/benchmarks/bench_calibration.py`
- `phantom-core/tests/audit_suite.py` (Validates Sections 1 to 8 of `PHANTOM_PLATFORM_BUILD_PROMPT.md`)

---

### Phase 12: Documentation, Packaging & Installers
**Target Files:**
- `phantom-core/README.md` (Updated master documentation)
- `phantom-core/pyproject.toml`
- `phantom-core/Cargo.toml` (Root workspace)
- `phantom-core/install.sh` (Linux/macOS/WSL2 installer)
- `phantom-core/install.ps1` (Windows PowerShell installer)
- `phantom-core/docs/ARCHITECTURE.md`
- `phantom-core/docs/INNOVATIONS.md`
- `phantom-core/docs/INSTALL.md`
- `phantom-core/docs/API.md`
- `phantom-core/docs/PHANTOMFILE.md`
- `phantom-core/docs/PLUGINS.md`
- `phantom-core/docs/OLLAMA_MIGRATION.md`
- `phantom-core/docs/GGUF_SUPPORT.md`

---

## Verification Plan

### Automated Verification:
1. **GGUF Dequantization Test**: Round-trip verification on synthetic and reference Q4_K_M/Q8_0/BF16 tensors with max error < 1e-4 and throughput ≥ 2 GB/s.
2. **Binary Format Test**: Write 5 `.phantomw` layers, parse back, and verify exact byte-for-byte fidelity.
3. **Phantomfile Test**: Parse valid Phantomfile, test Ollama Modelfile compatibility, verify rejection of invalid directives.
4. **Gateway Test**: Test Bearer auth (401 vs 200), rate limiting (429 with Retry-After), request logging in JSONL, Ollama compatibility endpoints (`/api/tags`, `/api/chat`, `/api/generate`).
5. **CLI Test**: Execute `phantom plan llama3:70b`, `phantom doctor`, `phantom list --json`, `phantom status`.
6. **Web Dashboard Build**: Verify `ui/web` builds cleanly with Vite into static assets served by the gateway.
7. **Audit Suite**: Run `python tests/audit_suite.py` to generate the complete Section 1–8 compliance report.

### Manual Verification:
- Run `phantom plan llama3:70b` and inspect the formatted terminal output.
- Start `phantom serve` and verify WebUI on `http://localhost:11411/ui`.
- Connect an Ollama-compatible client (or curl) to `http://localhost:11411/api/chat`.
