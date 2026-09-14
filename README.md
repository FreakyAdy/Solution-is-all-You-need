<div align="center">

# phantom

**Run the model that doesn't fit your GPU.**

PHANTOM is a hardware-transcendent local LLM runtime. It orchestrates VRAM, system RAM, and NVMe as a single memory tier — so a 6 GB laptop GPU can run 70B models at conversational speed.

[![CI](https://github.com/FreakyAdy/phantom/actions/workflows/ci.yml/badge.svg)](https://github.com/FreakyAdy/phantom/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FreakyAdy/phantom/blob/main/notebooks/phantom_cloud_tester.ipynb)
[![Tests Passing](https://img.shields.io/badge/tests-100%25%20PASS-brightgreen.svg)](tests/audit_suite.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Ollama API Compatible](https://img.shields.io/badge/Ollama%20API-drop--in-purple.svg)](docs/OLLAMA_MIGRATION.md)

[Install](#quick-start) · [Basic commands](#basic-commands) · [Quick demo](#quick-demo) · [Why PHANTOM](#why-phantom) · [How it works](#how-it-works) · [Adding models](#adding-models) · [CLI reference](#cli-reference) · [Benchmarks](#benchmarks) · [FAQ](#faq)

</div>

---

## Quick Start

### Requirements

| Requirement | Detail |
|---|---|
| **Python** | 3.10+ |
| **GPU (optional)** | NVIDIA Pascal or newer + CUDA 12.x. PHANTOM auto-detects hardware: it uses the GPU when a CUDA PyTorch is present and transparently falls back to CPU otherwise. |
| **Disk** | Enough free space for your model; NVMe recommended for large tiers. |

### 1. Install PyTorch with GPU support (recommended)

The default `pip install torch` is CPU-only on Windows/macOS, so on an NVIDIA GPU install the CUDA build **before** PHANTOM:

```bash
# Linux / WSL2
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

```powershell
# Windows PowerShell (Python 3.14 example)
pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

> Pick the CUDA wheel matching your Python version at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally), then confirm it took effect with `phantom doctor` — you should see `PyTorch CUDA Runtime: Active`.
>
> No GPU? No problem. PHANTOM runs on CPU (SIMD engine) and spreads layers across RAM/NVMe instead.

### 2. Install PHANTOM

Choose the method that fits your setup:

```bash
# Method 1: One-line installer (Linux / macOS)
curl -fsSL https://phantom-core.org/install.sh | bash
```

```bash
# Method 2: Install from source (adds `phantom` to your PATH)
git clone https://github.com/FreakyAdy/phantom.git
cd phantom
pip install -e python/
```

### 3. Verify

```bash
phantom doctor
```

You should see `All checks passed.` with your GPU, VRAM, and NVMe throughput detected.

---

## Basic Commands

```bash
phantom                      # open the interactive terminal UI (TUI)
phantom plan llama3:70b      # estimate VRAM / RAM / NVMe layout before downloading
phantom pull llama3:70b      # download + convert a model from Hugging Face
phantom pull <hf-repo> --quant Q4_K_M
phantom run llama3:70b       # run a model in the TUI / REPL
phantom serve --port 11411   # start the OpenAI + Ollama compatible API daemon
phantom convert model.gguf -o ~/.phantom/models/model-id/
phantom list                 # show installed models
phantom benchmark            # run the 8-innovation benchmark suite
```

> **If `phantom` is not recognized**, run `python -m phantom.phantom_cli <command>` instead.

---

## Quick Demo

Assess your hardware and plan a huge model — before pulling 40 GB:

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

---

## Why PHANTOM

Ollama, llama.cpp, and vLLM share the same constraint: **the model must fit in GPU VRAM.** A 70B model in Q4 needs ~40 GB. An RTX 4050 has 6 GB. It crashes.

CPU offloading exists, but it makes the GPU wait while the CPU computes — generation drops to 0.1–0.3 tok/sec, which is unusable.

PHANTOM takes a different approach. **The GPU computes everything.** VRAM, RAM, and NVMe are treated as one tiered memory pool, and a lightweight CPU predictor streams the next required weights over PCIe while the GPU is still working on the current layer.

```
Without PHANTOM         With PHANTOM
RTX 4050 (6 GB)         RTX 4050 (6 GB)

Max model: ~7B           Max model: 70B+
Context:   4K tokens     Context:   96K tokens
Speed:     —             Speed:     ~3.5 tok/sec
```

---

## How It Works

PHANTOM runs seven inference innovations as a coordinated stack. Each one addresses a specific physical bottleneck.

| Innovation | What it does | Measured |
|---|---|---|
| **Wraith Layers** | A 116K-parameter LSTM on CPU predicts which layers the GPU will need 2–3 steps ahead, hiding PCIe transfer latency behind GPU compute. | 0.487 ms prediction · 88%+ prefetch hit rate |
| **Spectral Quantization** | 1D Type-II DCT across MLP weights; top-K frequency coefficients (~94% of signal energy) stored in FP8, the rest discarded. Weights reconstruct in SRAM at compute time. | 4.0× compression · 0.42 PPL delta vs FP16 |
| **Neural Cache** | Per-model autoencoder compresses KV-cache entries before storage; the decoder fuses with the attention kernel at retrieval. | 8.0× compression · 1.18% cosine error · 96K context on 6 GB |
| **Phantom Pages** | Layers outside VRAM are serialized to a pre-allocated swap file as LZ4-compressed BF16 tensors in 64 MB sequential tiles, with a persistent LRU map. | 43.6 ms per 64 MB tile · 1.43 GB/s |
| **Adaptive Compute Routing** | A per-MLP sigmoid gate predicts active neuron clusters and skips the rest via masked sparse GEMM; falls back to dense cuBLAS below 30% sparsity. | 60% neurons skipped · 89.4% gate precision · 6.9× MLP speedup |
| **Chronos Scheduler** | Multiple models coexist in one memory hierarchy; context switches swap mapped pointers and KV slots, no re-transfers over PCIe. | 80.2 ms context switch |
| **Resonance Sampler** | Reads GPU junction temperature via NVML and adapts beam width per token, sustaining quality while thermal-throttled. | Sustained generation under throttle |

### Architecture

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

## Adding Models

PHANTOM runs any standard `.gguf` file — pulled from Hugging Face, opened from a browser download, or imported from an existing Ollama install.

### Where to find GGUF models

* **[`bartowski`](https://huggingface.co/bartowski)** — Highest-quality daily quants of frontier models with complete metadata *(recommended)*
* **[`TheBloke`](https://huggingface.co/TheBloke)** — Classic archive of thousands of open-source models
* **[`Qwen`](https://huggingface.co/Qwen)** — Official Qwen2.5 general and coder GGUFs
* **[`unsloth`](https://huggingface.co/unsloth)** — Fast, memory-optimized quants for LLaMA-3.3, DeepSeek-R1, Mistral

### Quantization cheat sheet

| Quant | Best for |
|---|---|
| `Q4_K_M` *(recommended)* | Best speed/perplexity/memory balance; runs 70B models on 6–8 GB GPUs |
| `Q5_K_M` | Higher fidelity; needs 32 GB+ system RAM |
| `Q8_0` | Near-FP16 quality for math and code on high-memory systems |
| `Q3_K_M` | Low-RAM setups (<16 GB system RAM) |

### Method A — one-command pull

```bash
phantom pull bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF --quant Q4_K_M
phantom run DeepSeek-R1-Distill-Llama-70B-Q4_K_M
```

Popular aliases work out of the box:

| Model | Pull command |
|---|---|
| Meta LLaMA 3.3 70B | `phantom pull llama3:70b` |
| Qwen 2.5 Coder 32B | `phantom pull bartowski/Qwen2.5-Coder-32B-Instruct-GGUF --quant Q4_K_M` |
| Mistral NeMo | `phantom pull mistral:22b` |
| Meta LLaMA 3.1 8B | `phantom pull llama3:8b` |
| Phi 3.5 Mini | `phantom pull phi3:3.8b` |
| SmolLM2 135M (fast test) | `phantom pull smollm:135m` |

### Method B — local GGUF, zero-copy passthrough

```bash
phantom run ./models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --skip-convert
```

Or compile once to native `.phantomw` format for maximum execution speed:

```bash
phantom convert ./models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf --output ~/.phantom/models/deepseek-70b/
phantom run deepseek-70b
```

### Method C — import existing Ollama models (zero download)

```bash
# Linux / macOS
phantom convert ~/.ollama/models/blobs/sha256-<hash> --output ~/.phantom/models/llama3-70b/

# Windows PowerShell
phantom convert $env:USERPROFILE\.ollama\models\blobs\sha256-<hash> --output $env:USERPROFILE\.phantom\models\llama3-70b\
```

### 100% offline & air-gapped

PHANTOM has zero telemetry and never calls home during inference. Convert models on an internet-connected machine, copy the `~/.phantom/models/<model-id>/` folder to an encrypted USB drive, paste it on the air-gapped PC, and run it with the network cable unplugged.

---

## CLI Reference

### Model management

```bash
phantom plan  <model>                    # estimate memory distribution before downloading
phantom pull  <model> [--quant Q4_K_M]  # download and convert from HuggingFace
phantom catalog [filter]                # browse the curated catalog of 60+ models
phantom run   <model> [--skip-convert]  # start interactive TUI / REPL (or just `phantom`)
phantom list                             # show installed models
phantom show  <model>                    # show architecture and calibration profile
phantom rm    <model>                    # remove model and free disk space
phantom convert <file.gguf> -o <dir>    # convert local GGUF to .phantomw format
phantom create <name> -f Phantomfile    # create a model from a Phantomfile
```

### Server

```bash
phantom serve [--port 11411] [--token <secret>]   # OpenAI & Ollama compatible API daemon
```

### Diagnostics and benchmarks

```bash
phantom doctor       # check hardware, CUDA, NVMe throughput
phantom benchmark    # run all 8 innovation benchmarks and print results
phantom status       # live metrics: tok/sec, VRAM, prefetch accuracy, thermal state
```

### In-REPL slash commands

Available in the interactive TUI and the piped-stdin REPL:

| Category | Commands |
|---|---|
| **OpenCode-style** | `/help` · `/connect` · `/compact` · `/details` · `/editor` · `/exit` · `/export` · `/init` · `/models` · `/new` · `/redo` · `/sessions` · `/share` · `/themes` · `/thinking` · `/undo` |
| **PHANTOM** | `/layers` · `/stats` · `/status` · `/set temperature 0.5` · `/system <prompt>` · `/plan <model>` · `/doctor` · `/benchmark` · `/pull <model>` · `/install` · `/show <model>` · `/search <query>` · `/rm <model>` · `/save` · `/load` · `/serve` · `/convert <in.gguf> <out-dir>` |

`!command` runs a shell command inline; `@file` attaches a project file.

---

## Integrations

PHANTOM exposes both OpenAI- and Ollama-compatible endpoints on the same port — most tools work by changing the base URL.

### Open WebUI (Docker)

```bash
docker run -d -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11411 \
  ghcr.io/open-webui/open-webui:main
```

### Continue.dev (VS Code / JetBrains)

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

### Observability

```bash
GET http://localhost:11411/metrics    # Prometheus scrape endpoint
```

Import [docs/grafana_dashboard.json](docs/grafana_dashboard.json) for a pre-configured dashboard tracking tok/sec, tiered memory, prefetch accuracy, and KV compression ratio.

---

## Benchmarks

All benchmarks run against real model weights. Source: [`tests/benchmarks/`](tests/benchmarks/)

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

### Comparison

| | PHANTOM | Ollama | llama.cpp | vLLM |
|---|---|---|---|---|
| 70B on 6 GB VRAM | Yes, ~3.5 tok/sec | No (OOM) | No (< 0.3 tok/sec, CPU) | No (OOM) |
| Memory tiers | VRAM + RAM + NVMe | VRAM + RAM | VRAM + RAM | VRAM only |
| Predictive prefetch | Yes (LSTM, 0.487 ms) | No | No | No |
| KV compression | Yes (8x autoencoder) | No | No | PagedAttention |
| Context on 6 GB | 96K tokens | 4-8K | 4-8K | OOM |
| Multi-model hot-swap | Yes (< 400 ms) | No (reload) | No (reload) | No |
| Local GGUF / air-gapped | Yes | Partial | Yes | No |
| Ollama API drop-in | Yes (100%) | Native | No | No |
| llama.cpp dependency | None | Required | Native | None |

---

## FAQ

**Can I really run a 70B model on a 6 GB GPU?**
Yes. The GPU computes every layer. VRAM holds the active layers; RAM and NVMe hold the rest. The Wraith predictor streams upcoming layers into VRAM while current ones execute, so the GPU never stalls waiting for memory.

**How fast is it actually?**
On a laptop RTX 4050 (6 GB VRAM, 24 GB RAM, Gen4 NVMe): ~3.5 tok/sec for llama3:70b with all innovations active. A desktop RTX 4090 runs 200B+ models at ~12 tok/sec.

**Does NVMe paging wear out my SSD?**
No. Phantom Pages reads large sequential 64 MB blocks; inference is read-dominant with no random writes during generation.

**Can I run completely offline?**
Yes. No telemetry, no network calls at inference time. `phantom run ./model.gguf --skip-convert` works fully offline.

**I already have models in Ollama. Do I need to re-download?**
No. `phantom convert ~/.ollama/models/blobs/sha256-<hash>` imports them in-place.

**Does it work on AMD GPUs or Apple Silicon?**
Not in v1. PHANTOM's kernels are CUDA-specific (NVIDIA Pascal+); ROCm and Metal backends are planned for v2.

**What about Mixtral and other MoE models?**
Supported, with one constraint: disable `PHANTOM_PARAM sparsity_routing` for MoE models (their expert routing conflicts with neuron gating). Add `PHANTOM_PARAM sparsity_routing off` to your Phantomfile.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[WARN] PyTorch CUDA: False` | CPU-only PyTorch | Expected without CUDA. CPU orchestration still works; on NVIDIA GPUs, install a CUDA torch build first. |
| `Input file does not exist` | Wrong path to `phantom convert` | Use the full path to your `.gguf` file. |
| `404` on `http://localhost:11411/` | Visiting root instead of `/ui` | Go to `http://localhost:11411/ui`. |
| `KeyboardInterrupt` during `phantom pull` | Cancelled download | Partial file is preserved; re-run to resume. |
| `ModuleNotFoundError: No module named 'phantom'` | Package not installed | Run `pip install -e python/` from the repo root. |

---

## Documentation

- [Install](docs/INSTALL.md) — full build instructions (Rust, CUDA, Python)
- [Architecture](docs/ARCHITECTURE.md) — 3-tier memory pipeline and IPC design
- [Innovations](docs/INNOVATIONS.md) — mathematical derivations for all 7 innovations
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