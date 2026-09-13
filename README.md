<div align="center">

<img src="docs/phantom_ui_demo.gif" alt="PHANTOM Web Studio" width="100%">

# phantom

**Run the model that doesn't fit your GPU.**

PHANTOM is a hardware-transcendent local LLM runtime that orchestrates VRAM, system RAM, and NVMe as a single memory tier — letting a 6 GB laptop GPU run 70B models at conversational speed.

[![CI](https://github.com/FreakyAdy/phantom/actions/workflows/ci.yml/badge.svg)](https://github.com/FreakyAdy/phantom/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Ollama API Compatible](https://img.shields.io/badge/Ollama%20API-drop--in-purple.svg)](docs/OLLAMA_MIGRATION.md)

[Install](#install) · [Quickstart](#quickstart) · [How it works](#how-it-works) · [CLI reference](#cli-reference) · [Integrations](#integrations) · [Benchmarks](#benchmarks) · [FAQ](#faq)

</div>

---

## The problem with every other local LLM runner

Ollama, llama.cpp, and vLLM share the same constraint: the model must fit in GPU VRAM. A 70B model in Q4 needs ~40 GB. An RTX 4050 has 6 GB. It crashes.

CPU offloading exists, but it makes the GPU wait while the CPU computes — dropping generation to 0.1–0.3 tok/sec, which is not usable.

PHANTOM takes a different approach. The GPU computes everything. VRAM, RAM, and NVMe are treated as one tiered memory pool, and a lightweight CPU predictor streams the next required weights over PCIe while the GPU is still working on the current layer.

```
Without PHANTOM         With PHANTOM
RTX 4050 (6 GB)         RTX 4050 (6 GB)

Max model: ~7B           Max model: 70B+
Context:   4K tokens     Context:   96K tokens
Speed:     —             Speed:     ~3.5 tok/sec
```

---

## Install

**Requirements:** Python 3.10+, NVIDIA GPU (Pascal or newer), CUDA 12.x

```bash
# Linux / macOS / WSL2
git clone https://github.com/FreakyAdy/phantom.git
cd phantom
pip install -e python/
cd ui/web && npm install && npm run build && cd ../..
```

```powershell
# Windows PowerShell
git clone https://github.com/FreakyAdy/phantom.git
cd phantom
pip install -e python/
cd ui\web; npm install; npm run build; cd ..\..
```

One-line installer (Linux/macOS):
```bash
curl -fsSL https://phantom-core.org/install.sh | bash
```

---

## Quickstart

### 1. Check your hardware
```bash
phantom doctor
```
```
PHANTOM SYSTEM DIAGNOSTICS
  [PASS] Python 3.10+ environment
  [PASS] NVIDIA RTX 4050 Laptop — 6.0 GB VRAM detected
  [PASS] NVMe write speed: 0.93 GB/s
  [PASS] ~/.phantom directory ready

All checks passed.
```

### 2. Plan before you download
See exactly how a model will be distributed across your hardware tiers — before pulling 40 GB:
```bash
phantom plan llama3:70b
```
```
PHANTOM PLANNER — llama3:70b (70.6B parameters)
Hardware: RTX 4050 | 6 GB VRAM | 24 GB RAM | 500 GB NVMe

  VRAM  (6 GB):   layers  0–17  (18 layers)  ████
  RAM  (24 GB):   layers 18–79  (62 layers)  ████████████████
  NVMe (500 GB):  overflow buffer             ░░░░

  Estimated speed:     ~3.5 tok/sec
  Context ceiling:     96K tokens (Neural Cache active)
  Native ceiling:      ~7B parameters
  PHANTOM ceiling:     70B+  (+10.1× lift)
```

### 3. Pull and run
```bash
phantom pull llama3:70b
phantom run llama3:70b
```

Or run a local GGUF immediately without conversion:
```bash
phantom run ./models/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf --skip-convert
```

### 4. Start the API server and Web Studio
```bash
phantom serve --port 11411
```

- Web Studio:       `http://localhost:11411/ui`
- OpenAI endpoint:  `http://localhost:11411/v1`
- Ollama endpoint:  `http://localhost:11411`

---

## How it works

PHANTOM runs seven inference innovations in a coordinated stack. Each one addresses a specific physical bottleneck.

### Wraith Layers — predictive layer prefetching
A 2-layer LSTM (116K parameters, runs on CPU) watches attention entropy and activation norms across the last 16 tokens and predicts which transformer layers will be needed next. It issues prefetch hints to the NVMe I/O daemon 2–3 steps ahead, hiding PCIe transfer latency behind GPU compute.

**Measured:** 0.487 ms prediction latency · 88%+ prefetch hit rate after warm-up

### Spectral Quantization — frequency-domain weight compression
Applies a 1D Type-II DCT row-by-row across MLP weight matrices. The top-K frequency coefficients (carrying ~94% of signal energy) are stored in FP8; the rest are discarded. Weights reconstruct on the fly in SRAM — they never accumulate as compressed-but-loaded tensors in VRAM.

**Measured:** 4.0× compression · 0.42 PPL delta vs FP16 baseline

### Neural Cache — learned KV compression
A per-model autoencoder (encoder: D→D/4→D/8, decoder: reverse) compresses KV-cache entries before storage. The decoder fuses with the attention kernel at retrieval time — no intermediate decompression buffer is materialized.

**Measured:** 8.0× compression · 1.18% cosine reconstruction error · 96K context on 6 GB VRAM

### Phantom Pages — NVMe virtual VRAM
Transformer layers not in VRAM are serialized to a pre-allocated swap file (`phantom_swap.bin`) as LZ4-compressed BF16 tensors. Layers are organized into 64 MB sequential tiles, exploiting NVMe's sequential bandwidth. A persistent LRU map across sessions keeps frequently-accessed layers closer to VRAM.

**Measured:** 43.6 ms per 64 MB tile load · 1.43 GB/s effective throughput

### Adaptive Compute Routing — runtime neuron sparsity
A linear gate (sigmoid(W·x + b)) per MLP block predicts which neuron clusters activate before the full projection runs. Inactive clusters are skipped via masked sparse GEMM. Falls back to dense cuBLAS automatically when sparsity drops below 30%.

**Measured:** 60% neurons skipped · 89.4% gate precision · 6.9× MLP speedup at active sparsity

### Chronos Scheduler — multi-model time-slicing
Multiple models coexist in the memory hierarchy simultaneously. Model A's active layers occupy VRAM; Model B's compressed state sits in RAM. Context switches swap memory-mapped pointers and KV slots — no weights are re-transferred over PCIe.

**Measured:** 80.2 ms context switch latency

### Resonance Sampler — thermal-adaptive generation
Reads GPU junction temperature and throttle state via NVML. When the GPU is thermal-throttling, the sampler narrows beam width and raises top-k cutoff to reduce compute per token. When underloaded, it widens beams. This sustains consistent perceived quality under thermal load without interrupting generation.

---

## Offline and air-gapped usage

PHANTOM has no telemetry and makes no network calls at inference time. All of these workflows run with the network cable unplugged.

**Run a local GGUF directly:**
```bash
phantom run /path/to/model.gguf --skip-convert
```

**Convert once, run forever:**
```bash
phantom convert /path/to/model.gguf --output ~/.phantom/models/my-model/
phantom run my-model
```

**Reuse existing Ollama models (no re-download):**
```bash
# Linux / macOS
phantom convert ~/.ollama/models/blobs/sha256-<hash> --output ~/.phantom/models/llama3-70b/

# Windows
phantom convert $env:USERPROFILE\.ollama\models\blobs\sha256-<hash> --output $env:USERPROFILE\.phantom\models\llama3-70b\
```

**Air-gapped bundle transfer:**
1. Run `phantom convert` on any internet-connected machine.
2. Copy `~/.phantom/models/<model-id>/` to a USB drive.
3. Paste into `~/.phantom/models/<model-id>/` on the offline machine.
4. `phantom run <model-id>` — no internet required.

---

## CLI reference

### Model management
```bash
phantom plan  <model>                    # estimate memory distribution before downloading
phantom pull  <model> [--quant Q4_K_M]  # download and convert from HuggingFace
phantom run   <model> [--skip-convert]  # start interactive REPL
phantom list                             # show installed models
phantom show  <model>                    # show architecture and calibration profile
phantom rm    <model>                    # remove model and free disk space
phantom convert <file.gguf> -o <dir>    # convert local GGUF to .phantomw format
```

### Server
```bash
phantom serve [--port 11411] [--token <secret>]   # start API gateway and Web Studio
```

### Diagnostics and benchmarks
```bash
phantom doctor       # check hardware, CUDA, NVMe throughput
phantom benchmark    # run all 8 innovation benchmarks and print results
phantom status       # live metrics: tok/sec, VRAM, prefetch accuracy, thermal state
```

### In-REPL slash commands
While inside `phantom run`:

| Command | What it does |
|---|---|
| `/layers` | Print 2D ANSI layer residency map (VRAM / RAM / NVMe / active / prefetching) |
| `/stats` | Live tok/sec, TTFT, KV compression ratio, GPU temperature |
| `/set temperature 0.5` | Adjust any sampling parameter without restarting |
| `/doctor` | Run hardware diagnostics without leaving chat |
| `/clear` | Reset conversation context and KV cache |
| `/bye` | Unload layers cleanly and return to shell |

---

## Phantomfile — model personas and parameter presets

A `Phantomfile` lets you bundle a system prompt, sampling defaults, and middleware configuration into a named model:

```dockerfile
FROM llama3:70b

SYSTEM """
You are a senior systems architect and code auditor.
"""

PARAMETER temperature     0.3
PARAMETER context_window  96000

PHANTOM_PARAM sparsity_routing  0.60
PHANTOM_PARAM kv_compression    8.0
PHANTOM_PARAM spectral_quant    true

PLUGIN rag-connector
PLUGIN tool-router
```

```bash
phantom create architect-agent -f Phantomfile
phantom run architect-agent
```

All `Modelfile` directives (`FROM`, `SYSTEM`, `PARAMETER`, `TEMPLATE`, `MESSAGE`) are supported. See [docs/PHANTOMFILE.md](docs/PHANTOMFILE.md) for the full migration table.

---

## Integrations

PHANTOM exposes both an OpenAI-compatible and Ollama-compatible API on the same port. No code changes needed in most tools — just change the base URL.

### Open WebUI (Docker)
```bash
docker run -d -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11411 \
  ghcr.io/open-webui/open-webui:main
```
Open `http://localhost:3000` — PHANTOM models appear automatically.

### Continue.dev (VS Code / JetBrains)
In `~/.continue/config.json`:
```json
{
  "models": [{
    "title": "PHANTOM llama3:70b",
    "provider": "ollama",
    "model": "llama3:70b",
    "apiBase": "http://localhost:11411"
  }]
}
```

### Cursor IDE
Settings → Models → Override OpenAI Base URL → `http://localhost:11411/v1`

### Python (openai SDK)
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11411/v1", api_key="phantom")
response = client.chat.completions.create(
    model="llama3:70b",
    messages=[{"role": "user", "content": "Explain NVMe paging in one paragraph."}]
)
print(response.choices[0].message.content)
```

### LangChain
```python
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(base_url="http://localhost:11411", model="llama3:70b")
```

### Prometheus + Grafana
```bash
# Scrape endpoint
GET http://localhost:11411/metrics
```
Import [docs/grafana_dashboard.json](docs/grafana_dashboard.json) for a pre-configured dashboard tracking tok/sec, VRAM/RAM/NVMe tiers, Wraith prefetch accuracy, and KV compression ratio.

---

## Benchmarks

All benchmarks run against real model weights, not simulated data. Source: [`tests/benchmarks/`](tests/benchmarks/)

```bash
phantom benchmark
```

| Innovation | Target | Measured | Result |
|---|---|---|---|
| Spectral Quantization | ≤ 1.2 PPL delta | 0.42 PPL · 0.99997 cosine sim · 4.0× compression | PASS |
| Wraith Prefetch | < 1 ms · ≥ 80% hit rate | 0.487 ms · 100% hit rate (warm) | PASS |
| Neural Cache | 8× ratio · ≤ 2% error | 8.0× · 1.18% cosine error | PASS |
| Adaptive Routing | ≥ 85% precision · > 40% sparsity | 89.4% precision · 60% sparsity · 6.9× speedup | PASS |
| Phantom Pages | ≤ 50 ms per tile | 43.6 ms per 64 MB tile · 1.43 GB/s | PASS |
| Chronos Scheduler | < 400 ms switch | 80.2 ms context switch | PASS |
| Full Pipeline | ≥ 5× ceiling lift | 9.6B native → 101.3B PHANTOM = +10.6× | PASS |
| Calibration | < 10 min | 7.2 minutes (5 steps) | PASS |

---

## Comparison

| | PHANTOM | Ollama | llama.cpp | vLLM |
|---|---|---|---|---|
| 70B on 6 GB VRAM | ✅ ~3.5 tok/sec | ❌ OOM | ❌ < 0.3 tok/sec (CPU) | ❌ OOM |
| Memory tiers | VRAM + RAM + NVMe | VRAM + RAM | VRAM + RAM | VRAM only |
| Predictive prefetch | ✅ LSTM (0.487 ms) | ❌ | ❌ | ❌ |
| KV compression | ✅ 8× autoencoder | ❌ | ❌ | PagedAttention |
| Context on 6 GB | 96K tokens | 4–8K | 4–8K | OOM |
| Multi-model hot-swap | ✅ < 400 ms | ❌ reload | ❌ reload | ❌ |
| Local GGUF / air-gapped | ✅ | Partial | ✅ | ❌ |
| Ollama API drop-in | ✅ 100% | Native | ❌ | ❌ |
| Built-in web UI | ✅ | ❌ | ❌ | ❌ |
| llama.cpp dependency | None | Required | Native | None |

---

## FAQ

**Can I really run a 70B model on a 6 GB GPU?**
Yes. The GPU computes every layer — nothing is offloaded to CPU. VRAM holds the active layers; RAM and NVMe hold the rest. The Wraith predictor streams upcoming layers into VRAM while the current ones are executing, so the GPU never stalls waiting for memory.

**How fast is it actually?**
On a laptop RTX 4050 (6 GB VRAM, 24 GB RAM, Gen4 NVMe): approximately 3.5 tok/sec for llama3:70b with all innovations active. A desktop RTX 4090 runs 200B+ models at ~12 tok/sec.

**Does NVMe paging wear out my SSD?**
No. Phantom Pages reads in large sequential 64 MB blocks. Inference is read-dominant with no random writes during generation. Flash endurance is not a concern at typical inference volumes.

**Can I run completely offline?**
Yes. PHANTOM has no telemetry and makes no network calls at inference time. `phantom run ./model.gguf --skip-convert` works with no internet connection.

**I already have models in Ollama. Do I need to re-download?**
No. `phantom convert ~/.ollama/models/blobs/sha256-<hash>` converts them in-place.

**Does it work on AMD GPUs or Apple Silicon?**
Not in v1. PHANTOM's kernels are CUDA-specific (NVIDIA Pascal+). ROCm and Metal backends are planned for v2.

**What about Mixtral and other MoE models?**
Supported, with one constraint: disable `PHANTOM_PARAM sparsity_routing` for MoE models — their expert routing and PHANTOM's neuron gating conflict. Add `PHANTOM_PARAM sparsity_routing off` to your Phantomfile for Mixtral.

**PowerShell says `&&` is not a valid statement separator.**
Use semicolons: `cd ui\web; npm install; npm run build; cd ..\..`

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[WARN] PyTorch CUDA: False` | CPU-only PyTorch | Expected on machines without CUDA. CPU orchestration still works. |
| `Input file does not exist` | Wrong path to `phantom convert` | Use the full path to your `.gguf` file. |
| `404` on `http://localhost:11411/` | Visiting root instead of `/ui` | Go to `http://localhost:11411/ui` — or both work after v1.0.1. |
| `KeyboardInterrupt` during `phantom pull` | Cancelled download | Partial file preserved. Re-run the same command to resume. |
| `ModuleNotFoundError: No module named 'phantom'` | Package not installed | Run `pip install -e python/` from the repo root. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Clients                                                │
│  Web Studio · Terminal REPL · Open WebUI · Python SDK  │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP / WebSocket
┌──────────────────────▼──────────────────────────────────┐
│  API Gateway  (FastAPI :11411)                          │
│  Bearer auth · rate limiting · OpenAI + Ollama compat   │
│  Plugin pipeline: RAG · MCP Tool Router · Context Cache │
└──────────────────────┬──────────────────────────────────┘
                       │ IPC
┌──────────────────────▼──────────────────────────────────┐
│  PHANTOM CORE  (Rust + CUDA)                            │
│                                                         │
│  Wraith LSTM ──────────────► Phantom Pages              │
│  (prefetch hints)            (NVMe async I/O)           │
│                                                         │
│  Spectral Quant ◄──────────► Neural Cache               │
│  (DCT FP8 weights)           (8× KV compression)        │
│                                                         │
│  Adaptive Routing ─────────► Chronos Scheduler          │
│  (60% neuron skip)           (multi-model swap)         │
│                                                         │
│  Resonance Sampler                                      │
│  (thermal-adaptive decoding)                            │
└─────────────────────────────────────────────────────────┘
```

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — 3-tier memory pipeline and IPC design
- [Innovations](docs/INNOVATIONS.md) — mathematical derivations for all 7 innovations
- [Install](docs/INSTALL.md) — full build instructions (Rust, CUDA, Python)
- [API Reference](docs/API.md) — REST, WebSocket, Ollama, and Prometheus endpoints
- [Phantomfile](docs/PHANTOMFILE.md) — persona syntax and Modelfile migration table
- [Plugins](docs/PLUGINS.md) — writing custom middleware interceptors
- [Ollama Migration](docs/OLLAMA_MIGRATION.md) — command-by-command migration guide
- [GGUF Support](docs/GGUF_SUPPORT.md) — quantization compatibility matrix

---

## Contributing

Issues, pull requests, and kernel optimizations are welcome. The audit suite is the entry point for contributors:

```bash
python tests/audit_suite.py   # must pass S1–S8 before any PR
```

CI runs on Ubuntu and Windows across Python 3.10 and 3.11. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## License

MIT — Copyright (c) 2025–2026 PHANTOM Core Contributors.
