# PHANTOM — Model Runtime Platform
### Build Prompt v1.0 | Extension of PHANTOM CORE
> *"Ollama runs the model that fits your GPU. PHANTOM runs the model that doesn't."*

---

## Table of Contents

1. [Prime Directive](#1-prime-directive)
2. [Feasibility Analysis](#2-feasibility-analysis)
3. [Architecture Overview](#3-architecture-overview)
4. [Component 1 — GGUF Native Loader](#4-component-1--gguf-native-loader)
5. [Component 2 — Model Format Conversion Pipeline](#5-component-2--model-format-conversion-pipeline)
6. [Component 3 — Model Lifecycle Manager](#6-component-3--model-lifecycle-manager)
7. [Component 4 — Phantomfile System](#7-component-4--phantomfile-system)
8. [Component 5 — CLI Interface](#8-component-5--cli-interface)
9. [Component 6 — Hardened API Gateway](#9-component-6--hardened-api-gateway)
10. [Component 7 — Plugin System](#10-component-7--plugin-system)
11. [Component 8 — Web Dashboard](#11-component-8--web-dashboard)
12. [Component 9 — Extended File Structure](#12-component-9--extended-file-structure)
13. [Performance Targets](#13-performance-targets--platform-layer)
14. [Ollama Migration Guide](#14-ollama-migration-guide)
15. [Audit & Validation Prompt](#15-audit--validation-prompt)
16. [Delivery Order](#16-delivery-order)
17. [Differentiation Summary](#17-differentiation-summary)

---

## 1. Prime Directive

**PHANTOM CORE** is already built: a hardware-transcendent LLM inference engine with 7 original innovations (Wraith Layers, Spectral Quant, Neural Cache, Phantom Pages, Adaptive Compute Routing, Chronos Scheduler, Resonance Sampler).

You are now building **PHANTOM** — the full model runtime platform that wraps PHANTOM CORE the same way Ollama wraps llama.cpp, but with:

1. A first-class model lifecycle manager (`pull` / `run` / `list` / `rm` / `serve`)
2. Native GGUF loading with automatic dequantize → Spectral Requantize pipeline
3. A Modelfile-equivalent system (**Phantomfile**) for model configuration and personas
4. A curated model library index backed by HuggingFace Hub and TheBloke/bartowski GGUF mirrors
5. A multi-model concurrent runtime with hot-swap (already partially built in Chronos)
6. A hardened OpenAI-compatible REST API with auth, rate-limiting, and request logging
7. A redesigned terminal UI + web dashboard showing what PHANTOM CORE uniquely enables
8. A plugin system for model middleware (system prompts, RAG connectors, tool call routing)

**New code lives entirely in the PHANTOM RUNTIME layer. PHANTOM CORE is unchanged except for:**
- Adding a format-agnostic weight loader interface
- Exposing the calibration pipeline as a callable Python API (not just CLI)

---

## 2. Feasibility Analysis

### The Gap Between Ollama and PHANTOM CORE

| Layer | Ollama | PHANTOM CORE (pre-extension) |
|---|---|---|
| Inference backend | llama.cpp | Custom Rust + CUDA |
| Model lifecycle | ✅ Full (`pull/run/list/rm`) | ❌ Missing |
| GGUF support | ✅ Native | ❌ Missing |
| OpenAI API compat | ✅ | ✅ (already built) |
| Multi-model coexistence | ❌ One at a time | ✅ Chronos Scheduler |
| >VRAM model support | ⚠️ Basic CPU offload | ✅ Full 3-tier hierarchy |
| Context beyond VRAM | ❌ | ✅ 8× via Neural Cache |
| Hardware-adaptive quality | ❌ | ✅ Resonance Sampler |
| Visual layer map | ❌ | ✅ Built |
| Calibration profiles | ❌ | ✅ Built |

### Genuine Advantages Over Ollama (post-extension)

- **Run models 3–10× larger than VRAM** — Wraith + Phantom Pages vs llama.cpp's basic offload
- **8× longer context** via Neural Cache KV compression
- **Multi-model coexistence** — Chronos holds multiple models simultaneously
- **Hardware-adaptive quality** — Resonance Sampler, nothing like this in Ollama
- **Per-model calibration profiles** — model-specific optimization vs one-size-fits-all
- **Full VRAM telemetry UI** — Ollama has zero visual tooling

### Genuine Challenges (be honest about these)

| Challenge | Severity | Mitigation |
|---|---|---|
| GGUF is non-negotiable for community adoption | Critical | Component 1 — GGUF Loader |
| Model library cold start (Ollama has thousands of curated models) | High | HF Hub integration + community index |
| Windows native support (io_uring is Linux-only) | High | IOCP fallback or document WSL2 requirement |
| CPU-only / Apple Silicon | Medium | Document clearly; ROCm stub for future |
| GGUF on top of spectral quant needs dequantize pipeline | High | Component 2 — Conversion Pipeline |

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PHANTOM CLI                              │
│              phantom pull / run / list / serve / plan           │
├─────────────────────────────────────────────────────────────────┤
│                      PHANTOM UI (Web)                           │
│            Dashboard · Models · Chat · Metrics · Layers         │
├─────────────────────────────────────────────────────────────────┤
│                    PHANTOM RUNTIME  ◄── NEW LAYER               │
│                                                                 │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐ │
│  │  Model Mgr   │  │  Phantomfile  │  │    Plugin System     │ │
│  │ pull/index   │  │   System      │  │  RAG · Tools · Cache │ │
│  └──────────────┘  └───────────────┘  └──────────────────────┘ │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐ │
│  │  GGUF Loader │  │ Format Conv.  │  │    API Gateway       │ │
│  │  + Dequant   │  │  Pipeline     │  │  Auth · RL · Ollama  │ │
│  └──────────────┘  └───────────────┘  └──────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                  PHANTOM CORE  ◄── EXISTING ENGINE              │
│                                                                 │
│   Wraith Layers  │  Spectral Quant  │  Neural Cache            │
│   Phantom Pages  │  Adaptive Compute│  Chronos  │  Resonance   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Component 1 — GGUF Native Loader

> **This is the single most critical component. Without it, zero community adoption.**

**File:** `python/phantom/loader/gguf_loader.py`

GGUF (GPT-Generated Unified Format) is a binary format storing model metadata, pre-quantized weight tensors, and tensor metadata. PHANTOM must load it without depending on llama.cpp.

### Class Interface

```python
class GGUFLoader:
    """
    Loads GGUF files and provides weight tensors to PHANTOM CORE.

    MODE A — PASSTHROUGH:
        Feed GGUF quantized weights directly to a quantization-aware inference
        path. Fastest — no conversion needed.
        Supports: Q4_K_M, Q5_K_M, Q8_0, F16

    MODE B — CONVERT:
        Dequantize GGUF weights to BF16, then apply Spectral Quantization.
        One-time conversion stored as .phantom file.
        Better compression + all PHANTOM CORE innovations.
        Recommended for models run repeatedly.

    Auto-selects mode:
        .phantom profile exists → PASSTHROUGH with calibration
        First run              → prompt user to choose mode
    """

    def __init__(self, gguf_path: Path, mode: Literal["passthrough", "convert", "auto"])

    def parse_header(self) -> GGUFMetadata
        """
        Parse GGUF header WITHOUT loading weights.
        Returns: model_type, num_layers, hidden_dim, num_heads, num_kv_heads,
                 rope_theta, context_length, tokenizer_type, quant_types_per_tensor
        """

    def load_tensor(self, name: str) -> torch.Tensor
        """
        Load a single named tensor using memory-mapped I/O.
        Dequantizes if necessary based on quant_type.
        Returns: BF16 tensor on CPU.
        """

    def iter_tensors(self) -> Iterator[Tuple[str, torch.Tensor]]
        """
        Iterate over all tensors in GGUF load order (not alphabetical).
        Uses memory-mapped file access — does NOT load entire model into RAM.
        """

    def dequantize_tensor(
        self, data: bytes, quant_type: GGUFQuantType, shape: Tuple
    ) -> torch.Tensor
        """
        Dequantize raw GGUF bytes to BF16.
        Must support:
          Q4_0, Q4_1, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M,
          Q6_K, Q8_0, Q8_1, F16, BF16, F32
        Implemented in pure PyTorch + numpy — no ctypes to llama.cpp.
        Reference: llama.cpp GGUF spec for bit packing format.
        """

    def detect_architecture(self) -> ModelArchitecture
        """
        From GGUF metadata, detect model family and return ModelArchitecture.
        Supported:
          LLaMA (1/2/3, 3.1, 3.2, 3.3)
          Mistral (0.1, 0.2, Mixtral 8x7B, 8x22B) — MoE flag detection required
          Gemma (1, 2)
          Qwen (1.5, 2, 2.5)
          Phi (2, 3, 3.5)
          Command-R, Command-R+
          DeepSeek (V2, V3)   — MLA attention detection required
          Falcon
        """
```

### GGUF Dequantization Requirements

- **Correctness:** Verify each quant type against llama.cpp reference. Assert max absolute error < 1e-4
- **Primary test case:** Q4_K_M — the most common community format
- **Memory:** Use `mmap` — do NOT read entire GGUF into RAM. A 70B GGUF is ~40GB
- **Access pattern:** Seek directly to each tensor's file offset using the GGUF index
- **Performance target:** ≥ 2 GB/s dequantization throughput (CPU, SIMD-accelerated via `torch.compile` or explicit AVX2)

---

## 5. Component 2 — Model Format Conversion Pipeline

**File:** `python/phantom/converter/phantom_convert.py`

**CLI:** `phantom convert --input model.gguf --output ~/.phantom/models/llama3-70b/`

### Conversion Pipeline

```
Step 1: Parse GGUF header → detect architecture
Step 2: For each tensor:
    a. Load raw GGUF tensor (memory mapped)
    b. Dequantize to BF16
    c. If MLP weight    → apply Spectral Quantization (DCT compress to FP8)
    d. If attn weight   → store as BF16 (attn weights don't benefit from spectral quant)
    e. If embedding     → store as BF16
    f. Write to .phantomw native format
Step 3: Run calibration pipeline (Steps A–E from PHANTOM CORE spec)
Step 4: Bundle weights + calibration profile into final model directory
```

### PHANTOM Native Format — Model Directory Structure

```
~/.phantom/models/<model-id>/
├── manifest.json              # model metadata, architecture, param count, source GGUF hash
├── config.toml                # PHANTOM CORE engine config for this model
├── tokenizer/
│   ├── tokenizer.json         # HF tokenizer format
│   └── special_tokens.json
├── weights/
│   ├── embed.bf16.bin         # embedding table
│   ├── layer_000.phantomw     # spectral-quantized layer weights
│   ├── layer_001.phantomw
│   └── lm_head.bf16.bin
└── profile/
    ├── calibration.phantom    # PHANTOM CORE calibration profile
    ├── wraith_init.pt         # pre-warmed Wraith LSTM
    └── hardware_profile.toml  # hardware tier this was calibrated on
```

### `.phantomw` Binary Format Specification (per-layer)

```
[4 bytes]   magic:                  "PHTW"
[4 bytes]   version:                1
[4 bytes]   layer_id
[4 bytes]   num_tensors

For each tensor:
  [64 bytes]  tensor_name           (null-padded UTF-8)
  [4 bytes]   rows
  [4 bytes]   cols
  [4 bytes]   k_coefficients_per_row  (0 = not spectral-quantized, stored as BF16)
  [4 bytes]   compressed_bytes
  [N bytes]   compressed data         (FP8 DCT coefficients OR raw BF16)
```

### Conversion Performance Requirements

| Metric | Target |
|---|---|
| Conversion time for 70B GGUF | ≤ 45 minutes |
| Peak RAM during conversion | ≤ 24 GB (process tensors one at a time) |
| Idempotency | Re-running same input → identical output |

**Progress display format:**
```
Downloading llama3:70b (38.4 GB) ███████████░░░░░░░░░ 54% @ 245 MB/s  ETA 2m31s
Converting  llama3:70b           ████████░░░░░░░░░░░░ 41% | Layer 45/80
Calibrating llama3:70b           ████████████████░░░░ 80% | Step D: Gates
✓ llama3:70b ready
  Native ceiling on this hardware:  ~30B parameters
  PHANTOM ceiling with this model:  70B @ estimated 4.2 tok/sec
```

---

## 6. Component 3 — Model Lifecycle Manager

**File:** `python/phantom/registry/model_manager.py`

### Class Interface

```python
class ModelManager:
    """
    Manages the local model library at ~/.phantom/models/
    Handles pull, list, remove, show, and search operations.
    """

    PHANTOM_INDEX_URL = "https://phantom-models.io/index.json"
    # Falls back to HuggingFace Hub when model is not in Phantom index

    def pull(self, model_ref: str, quantization: str = "Q4_K_M") -> PullResult
    def list(self, format: Literal["table", "json"]) -> List[ModelInfo]
    def show(self, model_id: str) -> ModelDetails
    def rm(self, model_id: str, force: bool = False) -> None
    def search(self, query: str) -> List[ModelSearchResult]
```

### `pull()` — Model Reference Formats

```
"llama3:70b"                                     → Phantom index lookup
"meta-llama/Meta-Llama-3-70B-Instruct"          → HuggingFace Hub
"/path/to/local.gguf"                            → Local file conversion
"https://example.com/model.gguf"                → Direct URL download
```

### `list()` — Table Output Format

```
NAME              SIZE    QUANT     CONTEXT    TOK/SEC   MODIFIED
llama3:70b        21 GB   SPECTRAL  128K       4.2       2 days ago
mistral:22b       8.4 GB  SPECTRAL  32K        11.3      1 week ago
phi3:3.8b         1.4 GB  SPECTRAL  128K       47.2      3 weeks ago
```

### Model Index Format (`phantom-models.io/index.json`)

```json
{
  "version": 1,
  "updated": "2025-01-15T00:00:00Z",
  "models": [
    {
      "id": "llama3:70b",
      "name": "Meta LLaMA 3 70B Instruct",
      "family": "llama3",
      "parameters": "70B",
      "context_length": 131072,
      "sources": {
        "Q4_K_M": {
          "url": "https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF/...",
          "sha256": "...",
          "size_bytes": 41234567890
        }
      },
      "community_profiles": [
        {
          "hardware_tier": "laptop_6gb",
          "calibrated_by": "phantom-community",
          "profile_url": "https://phantom-models.io/profiles/llama3-70b-laptop.phantom",
          "wraith_accuracy": 87.3,
          "kv_ratio": 7.8,
          "tok_per_sec": 3.8
        }
      ],
      "phantomfile": "FROM llama3:70b\nSYSTEM You are a helpful assistant.\n"
    }
  ]
}
```

---

## 7. Component 4 — Phantomfile System

**Phantomfile** is PHANTOM's equivalent of Ollama's Modelfile — a declarative configuration for named model personas.

**File:** `python/phantom/phantomfile/parser.py`

### Full Phantomfile Syntax Specification

```dockerfile
# Phantomfile
# -----------
# Create a custom model configuration from a base model.
# Usage: phantom create <name> -f Phantomfile

FROM <model-id>         # Required. Base model.
                        # Examples: llama3:70b, mistral:22b, phi3:3.8b

SYSTEM <text>           # System prompt (replaces model default)
SYSTEM """
Multi-line system prompts
are supported with triple quotes.
"""

TEMPLATE <jinja2>       # Chat template override (uses model default if omitted)
                        # Variables: {{ .System }}, {{ .Prompt }}, {{ .Response }}

# ── SAMPLING PARAMETERS ───────────────────────────────────────────────────────
PARAMETER temperature     0.7
PARAMETER top_p           0.9
PARAMETER top_k           40
PARAMETER repeat_penalty  1.1
PARAMETER seed            42       # -1 for random
PARAMETER num_predict     2048     # max tokens to generate
PARAMETER context_window  32768    # context window (up to model max)
PARAMETER stop            "<|eot_id|>"    # stop tokens (multiple allowed)
PARAMETER stop            "</s>"

# ── PHANTOM-SPECIFIC PARAMETERS (no Ollama equivalent) ───────────────────────
PHANTOM_PARAM sparsity_routing  on      # enable/disable sparse compute routing
PHANTOM_PARAM kv_compression    on      # enable/disable KV autoencoder
PHANTOM_PARAM spectral_quant    on      # enable/disable spectral quantization
PHANTOM_PARAM safe_mode         off     # lossless-only mode (disables lossy innovations)
PHANTOM_PARAM tier_preference   vram    # prefer vram | ram | nvme for hot layers
PHANTOM_PARAM max_vram_mb       4096    # hard VRAM cap (for GPU sharing)

# ── PLUGINS ──────────────────────────────────────────────────────────────────
PLUGIN rag-connector
PLUGIN tool-router

# ── METADATA ─────────────────────────────────────────────────────────────────
LICENSE MIT

# ── PRE-LOADED CONVERSATION ──────────────────────────────────────────────────
MESSAGE user      "Hello!"
MESSAGE assistant "Hi! How can I help?"
```

### Parser Class Interface

```python
class PhantomfileParser:

    def parse(self, path: Path) -> PhantomfileConfig

    def validate(self, config: PhantomfileConfig) -> List[ValidationError]

    def build_model(self, config: PhantomfileConfig, base_model: LoadedModel) -> PhantomModel
        """
        Applies Phantomfile config to a loaded base model:
        - Injects system prompt into the generation pipeline
        - Sets sampling parameters as defaults (overridable per-request)
        - Applies PHANTOM_PARAM settings to engine config
        - Loads specified plugins
        - Validates context_window <= model maximum
        """

    def to_modelcard(self, config: PhantomfileConfig) -> str
        """Generate a human-readable model card from a Phantomfile."""
```

### Ollama Modelfile → Phantomfile Migration

| Modelfile Directive | Phantomfile Equivalent | Notes |
|---|---|---|
| `FROM` | `FROM` | Identical |
| `SYSTEM` | `SYSTEM` | Identical |
| `TEMPLATE` | `TEMPLATE` | Identical |
| `PARAMETER` | `PARAMETER` | Identical |
| `MESSAGE` | `MESSAGE` | Identical |
| `LICENSE` | `LICENSE` | Identical |
| `ADAPTER` | Not supported (v1) | LoRA adapters — planned for v1.1 |
| *(none)* | `PHANTOM_PARAM` | New — PHANTOM-specific controls |
| *(none)* | `PLUGIN` | New — middleware plugins |

---

## 8. Component 5 — CLI Interface

**File:** `core/src/cli/main.rs` (extends existing `main.rs`)

Every command supports `--json` for machine-readable output.

### Full Command Reference

```
USAGE: phantom <command> [options]

──────────────────────────────────────────────────────────────────
MODEL MANAGEMENT
──────────────────────────────────────────────────────────────────

  phantom pull <model>
    Download and convert a model.
    --quant <type>          Quantization to download (default: Q4_K_M)
    --no-calibrate          Skip calibration (fast, reduced performance)
    --skip-convert          Use GGUF passthrough mode (no spectral quant)

  phantom run <model> [prompt]
    Run a model interactively or with a single prompt.
    --stream                Stream tokens to stdout
    --system <text>         Override system prompt
    --format json           Force JSON output mode
    --nowordwrap            Disable word wrapping
    -p, --parameter <k>=<v> Override any sampling parameter

  phantom list
    List all locally available models (table format).
    --json                  Output as JSON array

  phantom show <model>
    Show model details including calibration profile stats.

  phantom rm <model>
    Remove a model from local library.
    --force                 Skip confirmation prompt

  phantom search <query>
    Search Phantom model index + HuggingFace Hub simultaneously.

  phantom create <name> -f <Phantomfile>
    Create a named model persona from a Phantomfile.

  phantom push <name>
    (Future v1.1) Push a Phantomfile to phantom-models.io.

──────────────────────────────────────────────────────────────────
SERVER
──────────────────────────────────────────────────────────────────

  phantom serve
    Start the API server (OpenAI-compatible + PHANTOM extensions).
    --host 0.0.0.0          Bind address (default: 127.0.0.1)
    --port 11411            Port (default: 11411)
    --max-concurrent <n>    Max simultaneous generations
    --auth-token <token>    Enable bearer token authentication

──────────────────────────────────────────────────────────────────
PHANTOM-SPECIFIC COMMANDS
──────────────────────────────────────────────────────────────────

  phantom calibrate <model>
    Re-run calibration for a model.
    --steps A,B,C,D,E       Run specific calibration steps only
    --force                 Force re-calibration even if profile exists

  phantom status
    Show runtime status and all metrics in terminal-formatted output.
    Equivalent to: GET /v1/metrics, formatted for humans.

  phantom plan <model>
    Estimate resources WITHOUT loading the model.
    --vram <MB>             Override detected VRAM
    --ram <GB>              Override detected RAM
    --nvme <GB>             Override detected NVMe

  phantom convert <file>
    Convert GGUF to PHANTOM native format.
    --output <dir>          Output directory
    --mode passthrough|convert

  phantom doctor
    Run full system diagnostics:
    • CUDA kernel correctness tests
    • NVMe I/O speed measurement
    • Calibration profile validation
    • VRAM availability check
    • API server health check

  phantom update
    Update model index from phantom-models.io.
```

### `phantom plan` Output Format

```
phantom plan llama3:70b

Model: llama3:70b (70.6B parameters)

┌─────────────────────────────────────────────────────────┐
│ LAYER DISTRIBUTION (estimated)                          │
│ VRAM  (6 GB):   layers  0–12   (13 layers)  ████       │
│ RAM  (32 GB):   layers 13–55   (43 layers)  ████████   │
│ NVMe(200 GB):   layers 56–79   (24 layers)  ████░░     │
└─────────────────────────────────────────────────────────┘

Estimated tok/sec:          3.8 (warm) / 1.2 (cold start)
Estimated calibration:      8m 20s
PHANTOM advantage:          +9.3× beyond native ceiling

Run it:  phantom pull llama3:70b && phantom run llama3:70b
```

### Interactive REPL Mode

```bash
$ phantom run llama3:70b
# No prompt → enters interactive mode

>>> what is the capital of france?
Paris.

>>> /system You are a pirate.
System prompt updated.

>>> /clear
Context cleared.

>>> /save session.json
Saved 3 turns to session.json.

>>> /load session.json
Loaded 3 turns.

>>> /stats
Speed: 4.2 tok/sec  |  KV: 8,192/32,768 tokens  |  Temp: 67°C  |  Sparsity: 61%

>>> /layers
[ASCII layer map rendered — see below]

>>> /exit
```

### `/layers` ASCII Layer Map

```
Layer Residency Map — llama3:70b  (80 layers)
████ VRAM  ████ RAM   ░░░░ NVMe   ▓▓▓▓ Active   ···· Prefetching

00–19:  ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ░░ ░░ ░░ ░░ ░░
20–39:  ▓▓ ·· ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░
40–59:  ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░
60–79:  ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░

Wraith prediction:   Next → layers [22, 23, 24]  (prefetching ···)
KV compression:      7.8×  |  Context: 16,384 / 32,768 tokens used
Active sparsity:     61% neurons skipped this token
Speed:               4.2 tok/sec  |  Thermal: nominal (67°C)
```

---

## 9. Component 6 — Hardened API Gateway

**File:** `python/phantom/api/gateway.py`

Extends `openai_compat.py` with production-grade features.

### Gateway Features

```python
class PhantomAPIGateway:
    """
    Production API gateway wrapping the OpenAI-compatible endpoint.

    Adds:
    ─ Bearer token authentication (optional, configured in config.toml)
    ─ Per-client rate limiting (token bucket per IP)
    ─ Request logging to ~/.phantom/logs/requests.jsonl
    ─ Request queue with configurable max depth
    ─ Graceful 503 rejection when queue is full (Retry-After header)
    ─ CORS support for web clients
    ─ Request/response size limits (default: 1MB request, 10MB response)
    """
```

### Full Endpoint Reference

#### OpenAI-Compatible Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `POST` | `/v1/completions` | Legacy completions |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/health` | Health check |
| `GET` | `/v1/metrics` | PHANTOM extension — full performance stats |

#### PHANTOM-Specific Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/phantom/models/<id>/profile` | Full calibration profile as JSON |
| `GET` | `/phantom/models/<id>/layers` | Current layer residency map |
| `POST` | `/phantom/models/<id>/pin-layer` | Force-pin a layer to a memory tier |
| `POST` | `/phantom/calibrate/<id>` | Trigger calibration, returns `job_id` |
| `GET` | `/phantom/calibrate/<job_id>` | Poll calibration status and progress |
| `GET` | `/phantom/hardware` | Detected hardware profile and tier |
| `WS` | `/phantom/metrics/stream` | WebSocket metrics at 200ms intervals |
| `POST` | `/phantom/convert` | Submit GGUF path for conversion |

#### Ollama Compatibility Endpoints (drop-in replacement)

| Method | Endpoint | Maps To |
|---|---|---|
| `POST` | `/api/generate` | `/v1/completions` |
| `POST` | `/api/chat` | `/v1/chat/completions` |
| `GET` | `/api/tags` | `/v1/models` |
| `POST` | `/api/pull` | trigger `phantom pull` |
| `DELETE` | `/api/delete` | trigger `phantom rm` |
| `POST` | `/api/show` | `/phantom/models/<id>/profile` |

> **The Ollama compat layer makes PHANTOM a drop-in replacement for any tool already integrating with Ollama** — Open WebUI, Continue.dev, Cursor, etc. can switch with a single config change.

### `/v1/metrics` Response Schema

```json
{
  "vram_mb": 5821,
  "ram_mb": 22400,
  "nvme_mb": 45000,
  "layer_residency": {
    "vram": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "ram": [13, 14, "...", 55],
    "nvme": [56, 57, "...", 79]
  },
  "wraith_accuracy_pct": 87.3,
  "kv_compression_ratio": 7.8,
  "active_sparsity_pct": 61.2,
  "tok_per_sec": 4.7,
  "thermal_state": "nominal",
  "throttle_active": false,
  "active_model": "llama3:70b",
  "context_tokens_used": 16384,
  "context_tokens_max": 32768,
  "queued_requests": 0
}
```

---

## 10. Component 7 — Plugin System

**File:** `python/phantom/plugins/base.py`

Plugins are Python packages installed into `~/.phantom/plugins/` that implement the `PhantomPlugin` protocol. They intercept at four points in the generation pipeline.

### Plugin Protocol

```python
class PhantomPlugin(Protocol):
    """
    Intercepts at four pipeline points:
      1. pre_request:   modify or reject the incoming API request
      2. pre_generate:  modify the prompt before generation starts
      3. on_token:      process each generated token (can inject/suppress)
      4. post_generate: process the completed response
    """

    name: str       # unique plugin identifier
    version: str    # semver

    async def pre_request(
        self, request: GenerateRequest
    ) -> GenerateRequest | Rejection

    async def pre_generate(
        self, prompt: str, context: GenerationContext
    ) -> str

    async def on_token(
        self, token: str, context: GenerationContext
    ) -> str | None

    async def post_generate(
        self, response: str, context: GenerationContext
    ) -> str
```

### Built-in Plugins

#### RAGPlugin — Retrieval-Augmented Generation

```
File:    python/phantom/plugins/rag_connector/
Activate: PLUGIN rag-connector  (in Phantomfile)

Config: ~/.phantom/plugins/rag/config.toml
  [rag]
  db_path           = "~/.phantom/rag/default"
  top_k             = 5
  embedding_model   = "nomic-embed-text"
  max_context_tokens = 2048

Behavior:
  1. Embed the query using a local embedding model
  2. Retrieve top-k chunks from ChromaDB or LanceDB
  3. Inject retrieved context into prompt before generation
```

#### ToolRouterPlugin — Function Calling

```
File:    python/phantom/plugins/tool_router/
Activate: PLUGIN tool-router  (in Phantomfile)

Supports:
  - Python function tools (decorated with @phantom_tool)
  - HTTP endpoint tools
  - MCP (Model Context Protocol) tool servers
```

#### ContextCachePlugin — Prefix KV Caching

```
File:    python/phantom/plugins/context_cache/
Activate: PLUGIN context-cache  (in Phantomfile)

Config: ~/.phantom/plugins/context-cache/config.toml
  [context-cache]
  max_cached_prefixes = 10
  min_prefix_tokens   = 128    # only cache if system prompt > 128 tokens

PHANTOM advantage:
  Stores compressed KV state (via Neural Cache) so the cache takes
  8× less VRAM than an uncached prefill state would require.
  Prefix re-use latency target: ≤ 100ms.
```

---

## 11. Component 8 — Web Dashboard

The primary UI is a self-hosted web dashboard served by the API gateway. Electron is kept as an optional desktop wrapper.

**Directory:** `ui/web/`

### Page Structure

```
ui/web/src/pages/
├── Dashboard.tsx    # Overview: hardware tier, running models, recent requests
├── Models.tsx       # Model library: pull, convert, rm, show with progress
├── Chat.tsx         # Built-in chat interface with PHANTOM stats sidebar
├── Metrics.tsx      # Real-time performance graphs (200ms refresh via WS)
├── Layers.tsx       # Visual layer residency map (existing LayerMap component)
└── Plugins.tsx      # Plugin management and configuration
```

### "Ceiling Lift" Hero Component (`CeilingLift.tsx`)

This is the centerpiece that communicates PHANTOM's core value proposition instantly.

```
Your Hardware:  RTX 4050 (6GB VRAM)  +  32GB RAM  +  500GB NVMe

  WITHOUT PHANTOM              WITH PHANTOM
  ┌──────────────┐             ┌────────────────────────────────────────┐
  │  7B max      │             │  70B+ capable                          │
  │  ████        │    →→→      │  ████████████████████████████████████  │
  │ native limit │             │           PHANTOM ceiling              │
  └──────────────┘             └────────────────────────────────────────┘

Currently running: llama3:70b
  VRAM  ████████████░░  5.8 / 6.0 GB    (layers 0–12)
  RAM   ████████░░░░░░  18.4 / 32.0 GB  (layers 13–55)
  NVMe  ████░░░░░░░░░░  22.1 / 500 GB   (layers 56–79)
  Speed: 4.2 tok/sec  ●  Wraith: 87%  ●  KV ratio: 7.8×  ●  Temp: 67°C
```

### LayerMap Component Visual Spec

```
Each cell = one transformer layer.
Color indicates residency tier:

  ● Bright amber    — VRAM (hot)
  ● Soft blue       — RAM (warm)
  ● Dark slate      — NVMe (cold)
  ● Pulsing green   — Currently executing
  ● Blinking yellow — Being prefetched by Wraith

Layout: 2D grid, sqrt(N) × sqrt(N) reading order.
Hover: shows layer_id, type (attn/mlp), size_mb, last_access_ms_ago
Click: force-pin layer to VRAM (persists until manually unpinned)
Refresh: every 200ms via WebSocket from /phantom/metrics/stream
```

---

## 12. Component 9 — Extended File Structure

```
phantom/                               # renamed from phantom-core/
│
├── core/                              # UNCHANGED — Rust inference engine
│
├── kernels/                           # UNCHANGED — CUDA kernels
│
├── python/
│   ├── phantom/
│   │   ├── [existing files]           # UNCHANGED
│   │   │
│   │   ├── loader/                    # NEW
│   │   │   ├── __init__.py
│   │   │   ├── gguf_loader.py         # GGUF native loader + dequantizer
│   │   │   ├── safetensors_loader.py  # HF safetensors support
│   │   │   └── format_detect.py       # Auto-detect model format from path/header
│   │   │
│   │   ├── converter/                 # NEW
│   │   │   ├── __init__.py
│   │   │   ├── phantom_convert.py     # GGUF → PHANTOM conversion pipeline
│   │   │   └── format_spec.py         # .phantomw binary format read/write
│   │   │
│   │   ├── registry/                  # NEW
│   │   │   ├── __init__.py
│   │   │   ├── model_manager.py       # pull, list, rm, show, search
│   │   │   ├── index_client.py        # phantom-models.io index client
│   │   │   ├── hf_client.py           # HuggingFace Hub integration
│   │   │   └── downloader.py          # Resumable downloader with SHA256 verify
│   │   │
│   │   ├── phantomfile/               # NEW
│   │   │   ├── __init__.py
│   │   │   ├── parser.py              # Phantomfile parser
│   │   │   └── validator.py           # Validation + error reporting
│   │   │
│   │   ├── plugins/                   # NEW
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Plugin protocol + dynamic loader
│   │   │   ├── rag_connector/         # Built-in RAG plugin
│   │   │   ├── tool_router/           # Built-in tool use / function calling plugin
│   │   │   └── context_cache/         # Built-in prefix KV caching plugin
│   │   │
│   │   └── api/
│   │       ├── openai_compat.py       # UNCHANGED
│   │       ├── gateway.py             # NEW — hardened gateway layer
│   │       ├── ollama_compat.py       # NEW — Ollama API compatibility layer
│   │       └── websocket_stream.py    # UNCHANGED
│   │
│   └── phantom_cli.py                 # NEW — CLI entry point
│
├── ui/
│   ├── web/                           # NEW — primary browser-based dashboard
│   └── electron/                      # RENAMED — optional desktop wrapper
│
├── phantom-models/                    # NEW — separate git submodule
│   └── index.json                     # Community model index
│
├── docs/
│   ├── [existing docs]                # UNCHANGED
│   ├── PHANTOMFILE.md                 # NEW — Phantomfile language reference
│   ├── PLUGINS.md                     # NEW — Plugin development guide
│   ├── OLLAMA_MIGRATION.md            # NEW — Migration guide for Ollama users
│   └── GGUF_SUPPORT.md               # NEW — GGUF compatibility matrix per quant type
│
└── install.sh                         # UPDATED — installs full PHANTOM platform
```

---

## 13. Performance Targets — Platform Layer

Additional targets for the PHANTOM runtime layer, on top of existing PHANTOM CORE targets.

### Platform Layer Targets

| Metric | Target |
|---|---|
| GGUF Q4_K_M dequantization speed | ≥ 2 GB/s (CPU, SIMD) |
| Cold model load from local `.phantom` | ≤ 30s for 70B on NVMe Gen4 |
| `phantom pull` conversion throughput (GGUF → PHANTOM) | ≤ 45 min for 70B |
| API gateway request overhead | ≤ 2ms per request |
| Plugin pipeline overhead (`on_token`) | ≤ 5ms per token |
| Ollama compat endpoint latency delta | ≤ 10ms vs native PHANTOM endpoint |
| Web dashboard time-to-interactive | ≤ 1.5s on localhost |
| Context-cache prefix hit restore time | ≤ 100ms |

### Inherited PHANTOM CORE Targets (must still pass)

| Metric | Target |
|---|---|
| Wraith prefetch accuracy | ≥ 80% |
| Spectral Quant perplexity delta | ≤ 1.2 PPL vs FP16 |
| KV Autoencoder reconstruction error | ≤ 2% cosine distance |
| Sparsity gate precision | ≥ 85% on active neurons |
| Calibration time (any model) | ≤ 10 minutes |
| Layer load from NVMe Gen4 | ≤ 50ms per layer |
| Chronos model context switch | ≤ 400ms |

---

## 14. Ollama Migration Guide

**File:** `docs/OLLAMA_MIGRATION.md`

This document is required for community adoption. It must be honest about limitations.

### API Compatibility Table

| Ollama Endpoint | PHANTOM Support | Notes |
|---|---|---|
| `POST /api/generate` | ✅ Full | Maps to `/v1/completions` |
| `POST /api/chat` | ✅ Full | Maps to `/v1/chat/completions` |
| `GET /api/tags` | ✅ Full | Maps to `/v1/models` |
| `POST /api/pull` | ✅ Full | Triggers `phantom pull` |
| `DELETE /api/delete` | ✅ Full | Triggers `phantom rm` |
| `POST /api/show` | ✅ Full | Returns PHANTOM profile |
| `POST /api/push` | 🔄 Planned v1.1 | Community profile sharing |
| `POST /api/copy` | ✅ Full | Copies model config |
| `POST /api/create` | ✅ Full | `phantom create` equivalent |
| `GET /api/ps` | ✅ Full | Running models |

### Switching Open WebUI to PHANTOM

Change one environment variable:
```bash
# Before (Ollama)
OLLAMA_BASE_URL=http://localhost:11434

# After (PHANTOM)
OLLAMA_BASE_URL=http://localhost:11411
```

### Switching Continue.dev to PHANTOM

In `~/.continue/config.json`:
```json
{
  "models": [{
    "title": "PHANTOM — llama3:70b",
    "provider": "ollama",
    "model": "llama3:70b",
    "apiBase": "http://localhost:11411"
  }]
}
```

### Importing Existing Ollama Models

```bash
# Find where Ollama stores its models
ls ~/.ollama/models/blobs/

# Import an existing GGUF directly
phantom convert ~/.ollama/models/blobs/<sha256> --output ~/.phantom/models/llama3-70b/
```

### What PHANTOM Cannot Do (That Ollama Can)

- **Apple Silicon (M-series Macs):** PHANTOM is NVIDIA-only. Metal backend is not planned for v1.
- **AMD GPUs (ROCm):** Not supported in v1. ROCm stubs are in `kernels/rocm/` as a contribution target.
- **Windows native (without WSL2):** `io_uring` requires Linux. WSL2 is the supported Windows path.
- **ARM CPUs without CUDA GPU:** CPU-only inference is not supported.

### What PHANTOM Can Do (That Ollama Cannot)

- Run `llama3:70b` on a 6GB VRAM laptop — Ollama requires ~48GB VRAM for the same
- Maintain 8× longer context in the same VRAM via Neural Cache KV compression
- Run `llama3:70b` and `phi3:3.8b` simultaneously (Chronos Scheduler)
- Adapt generation quality to GPU thermal state (Resonance Sampler)
- Show a live visual map of where every model layer lives in memory
- Use model-specific calibration profiles for optimal performance per device

---

## 15. Audit & Validation Prompt

Use this prompt in a fresh agent session (Claude Code, Cursor, etc.) to validate the entire PHANTOM platform after building it.

---

### SECTION 1 — GGUF Loader Correctness

```
GGUF LOADER AUDIT

For each of the following quantization types, test dequantization correctness:
  Q4_0, Q4_1, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0, F16, BF16

For each type:
  1. Load a test tensor from a reference GGUF file using phantom GGUFLoader
  2. Load the same tensor using the llama.cpp Python bindings (reference)
  3. Assert: max(abs(phantom_output - reference_output)) < 1e-4
  4. Report: [PASS] or [FAIL: max_error=X, at_index=Y]

PRIMARY TEST CASE: Q4_K_M on a 4096×14336 weight matrix from llama3-8b.gguf

PERFORMANCE TEST:
  Time dequantization of a full 7B model's weights (all tensors).
  Report: total_time_seconds, throughput_GB_per_second
  Target: ≥ 2 GB/s
  [PASS] or [FAIL: actual=X GB/s]

MEMORY TEST:
  Monitor peak RAM during dequantization of a 70B GGUF.
  Verify: peak RAM stays under 24 GB (memory-mapped access, not full load).
  [PASS] or [FAIL: peak_ram=X GB]
```

---

### SECTION 2 — Conversion Pipeline

```
CONVERSION PIPELINE AUDIT

Run: phantom convert llama3-8b.gguf --output ~/.phantom/models/llama3-8b-test/

Verify:
  [ ] manifest.json is valid JSON with correct architecture fields
  [ ] All layer_NNN.phantomw files have correct PHTW magic bytes
  [ ] Calibration profile was generated (profile/calibration.phantom exists)
  [ ] Conversion is idempotent (run twice, output files are byte-identical)

TIME AUDIT:
  Time the full conversion pipeline on a 70B model.
  Report: [download_min] + [convert_min] + [calibrate_min] = [total_min]
  Target: ≤ 45 minutes total (excluding download time)
  [PASS] or [FAIL: actual=X min]

ROUND-TRIP CORRECTNESS:
  Load original GGUF weights (dequantized) and PHANTOM converted weights.
  For MLP layers: cosine_similarity(original_BF16, reconstructed_from_spectral_quant)
  Target: ≥ 0.995 mean cosine similarity across all MLP layers
  [PASS] or [FAIL: mean_cosine=X]
```

---

### SECTION 3 — CLI Interface

```
CLI AUDIT

Run each command and verify output:

  phantom doctor
    [ ] All kernel correctness checks pass
    [ ] NVMe speed measurement matches expected tier (Gen3/Gen4/Gen5)
    [ ] Returns exit code 0 on healthy system

  phantom plan llama3:70b
    [ ] Output shows layer distribution across VRAM/RAM/NVMe
    [ ] Estimated tok/sec is in a reasonable range (1–50 tok/sec)
    [ ] Does NOT load the model to produce this output
    [ ] Completes in < 2 seconds

  phantom pull phi3:3.8b (use a small model for CI speed)
    [ ] Download shows progress bar with speed and ETA
    [ ] Conversion shows per-layer progress
    [ ] Calibration shows per-step progress
    [ ] Final output shows ceiling lift summary

  phantom list
    [ ] Table renders correctly for N models
    [ ] --json flag returns valid JSON array

  phantom run phi3:3.8b "What is 2+2?"
    [ ] Returns correct answer ("4")
    [ ] --stream flag produces token-by-token output to stdout

  phantom serve + curl test:
    curl -X POST http://localhost:11411/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"phi3:3.8b","messages":[{"role":"user","content":"Hi"}]}'
    [ ] Returns valid OpenAI-format JSON response
    [ ] "model" field in response matches requested model

  phantom serve + Ollama compat test:
    curl -X POST http://localhost:11411/api/chat \
      -d '{"model":"phi3:3.8b","messages":[{"role":"user","content":"Hi"}]}'
    [ ] Returns valid Ollama-format JSON response

  phantom status
    [ ] Shows all metrics (VRAM, RAM, NVMe, tok/sec, thermal, Wraith accuracy)
    [ ] Refreshes without requiring a running generation
```

---

### SECTION 4 — Phantomfile System

```
PHANTOMFILE AUDIT

Create a test Phantomfile:
  FROM phi3:3.8b
  SYSTEM You are a helpful pirate assistant.
  PARAMETER temperature 0.5
  PHANTOM_PARAM kv_compression on
  PHANTOM_PARAM spectral_quant on

Run: phantom create test-pirate -f Phantomfile

Verify:
  [ ] phantom list shows "test-pirate" as an available model
  [ ] phantom run test-pirate "Hello" — response uses pirate persona
  [ ] temperature is 0.5 (verify via /v1/metrics or logging)
  [ ] kv_compression is active (verify via /v1/metrics kv_compression_ratio > 1)

OLLAMA MODELFILE IMPORT TEST:
  Take an existing Ollama Modelfile (any format-compatible one).
  Run: phantom create imported-model -f <OllamaModelfile>
  [ ] Import succeeds without manual editing
  [ ] OR parser clearly identifies which lines are incompatible (not a silent failure)

VALIDATION TEST:
  Create a Phantomfile with invalid values:
    FROM nonexistent:model
    PARAMETER temperature 999
    PARAMETER context_window 9999999
  Run: phantom create bad-model -f BadPhantomfile
  [ ] Returns helpful validation errors, not a crash
  [ ] Does NOT partially create the model
```

---

### SECTION 5 — Plugin System

```
PLUGIN AUDIT

RAG CONNECTOR:
  Setup: Index 10 text documents into ~/.phantom/rag/default
  Create: Phantomfile with PLUGIN rag-connector
  Run: phantom run rag-model "What does document 3 say about X?"
  [ ] Retrieved context appears in the generated response
  [ ] No documents retrieved for unrelated questions (precision check)
  [ ] Plugin latency: time(with_rag) - time(without_rag) < 200ms per request

CONTEXT CACHE:
  Create: Phantomfile with long SYSTEM prompt (500 tokens) + PLUGIN context-cache
  Run first request: note time-to-first-token (TTFT)
  Run second identical request: note TTFT
  [ ] Second request TTFT ≤ 100ms (prefix cached, no re-prefill)
  [ ] VRAM usage with cached prefix < VRAM usage without cache (8× compression)

TOOL ROUTER:
  Define a simple tool:
    @phantom_tool
    def get_weather(city: str) -> str:
        return f"Sunny in {city}"
  Run: phantom run tool-model "What's the weather in Tokyo?"
  [ ] Model calls the tool correctly
  [ ] Tool result is incorporated into the response
  [ ] No raw JSON tool-call syntax leaks into the user-visible response
```

---

### SECTION 6 — API Gateway

```
API GATEWAY AUDIT

AUTHENTICATION:
  Start: phantom serve --auth-token mysecrettoken
  Test 1: curl without token → expect 401 Unauthorized
  Test 2: curl with wrong token → expect 401 Unauthorized
  Test 3: curl with correct Bearer token → expect 200
  [ ] All three cases behave correctly

RATE LIMITING:
  Send 20 rapid requests from the same IP.
  [ ] First N requests succeed (where N = configured limit)
  [ ] Excess requests return 429 Too Many Requests with Retry-After header

CONCURRENT REQUEST HANDLING:
  Send 3 simultaneous chat completion requests.
  [ ] All 3 complete successfully (queued if max-concurrent=1)
  [ ] Responses are correctly matched to their respective requests
  [ ] No response mixing between concurrent requests

REQUEST LOGGING:
  Send 5 requests.
  Check: cat ~/.phantom/logs/requests.jsonl
  [ ] File has 5 entries
  [ ] Each entry has: timestamp, model, prompt_tokens, completion_tokens, duration_ms, client_ip

OLLAMA DROP-IN TEST:
  Point Open WebUI at http://localhost:11411
  [ ] Model list loads correctly
  [ ] Chat works end-to-end
  [ ] Streaming responses work
```

---

### SECTION 7 — Web Dashboard

```
WEB DASHBOARD AUDIT

LOAD TEST:
  Open http://localhost:11411/ui in a browser
  [ ] Page renders in < 1.5s
  [ ] No console errors in browser devtools

LAYER MAP:
  Load a model and start a generation.
  [ ] Layer map shows correct VRAM/RAM/NVMe distribution
  [ ] Active layer pulses green during generation
  [ ] Prefetching layers blink yellow
  [ ] Map updates within 200ms of layer state changes

METRICS PAGE:
  [ ] tok/sec graph updates in real-time during generation
  [ ] VRAM/RAM/NVMe gauges are accurate (cross-check with `phantom status`)
  [ ] Thermal state shows correct temperature

MODEL MANAGER:
  [ ] Pull a model via the UI (not CLI) — progress shows correctly
  [ ] Remove a model via the UI — it disappears from the list
  [ ] Show model details — calibration profile stats render correctly

CEILING LIFT COMPONENT:
  [ ] Shows correct native ceiling for detected hardware
  [ ] Shows correct PHANTOM ceiling for loaded model
  [ ] Numbers are not hardcoded — they update based on the actual detected hardware
```

---

### SECTION 8 — Open Source Readiness

```
OSS READINESS AUDIT

DOCUMENTATION:
  [ ] README.md has: one-sentence pitch, install command, quickstart in < 5 steps
  [ ] PHANTOMFILE.md covers all Phantomfile directives with examples
  [ ] OLLAMA_MIGRATION.md is accurate (test the config changes manually)
  [ ] GGUF_SUPPORT.md has a correct compatibility matrix

MISSING CRITICAL FEATURES (check each):
  [ ] Community profile registry implemented (or documented as roadmap)
  [ ] phantom convert tool implemented and tested
  [ ] Windows WSL2 documented and install.sh detects WSL2 correctly
  [ ] Mixtral MoE architecture handled correctly (MoE layers must not use
      standard sparsity routing — they have their own expert selection)
  [ ] DeepSeek V2/V3 MLA attention detected and handled
  [ ] phi3:3.8b works end-to-end (good small model for CI)

LICENSE:
  [ ] LICENSE file present in repo root
  [ ] All third-party CUDA code (FlashAttention variants) is license-compatible
  [ ] HuggingFace Hub client usage complies with their ToS

SECURITY:
  [ ] No hardcoded secrets in any config file
  [ ] API auth token is stored in config, not embedded in URL
  [ ] phantom_swap.bin is not world-readable (chmod 600 after creation)
  [ ] Input size limits prevent OOM from malicious large prompts
```

---

### Final Audit Output Format

```
PHANTOM PLATFORM AUDIT REPORT
Generated: [timestamp]
Phantom Version: [git hash]
Hardware: [GPU] [VRAM]GB VRAM + [RAM]GB RAM + [NVMe speed]

SECTION RESULTS:
  S1 GGUF Loader:         [X/Y PASS, Z FAIL]
  S2 Conversion Pipeline: [X/Y PASS, Z FAIL]
  S3 CLI Interface:       [X/Y PASS, Z FAIL]
  S4 Phantomfile:         [X/Y PASS, Z FAIL]
  S5 Plugin System:       [X/Y PASS, Z FAIL]
  S6 API Gateway:         [X/Y PASS, Z FAIL]
  S7 Web Dashboard:       [X/Y PASS, Z FAIL]
  S8 OSS Readiness:       [X/Y PASS, Z FAIL]

OVERALL: [SHIP IT / NEEDS WORK / BLOCKED]

CRITICAL BLOCKERS:
  [list any FAIL items that prevent basic function]

HIGH PRIORITY:
  [list FAIL items that affect core claims]

MEDIUM:
  [list WARN items]
```

---

## 16. Delivery Order

Build in this exact sequence. Each phase must compile and pass tests before moving forward.

```
PHASE 1   GGUF Loader
            gguf_loader.py — correctness tested against llama.cpp output
            Test: Q4_K_M round-trip on llama3-8b.gguf

PHASE 2   PHANTOM Native Format
            format_spec.py — .phantomw writer + reader
            Test: write 5 layers, read back, verify byte-identical

PHASE 3   Conversion Pipeline
            phantom_convert.py — tested end-to-end on llama3-8b GGUF
            Test: round-trip cosine similarity ≥ 0.995

PHASE 4   Model Manager
            model_manager.py — pull from HuggingFace Hub
            Test: phantom pull phi3:3.8b completes successfully

PHASE 5   Phantomfile Parser + Validator
            Test: parse a valid Phantomfile, reject an invalid one

PHASE 6   CLI Interface
            phantom run / pull / list / show / serve / plan / doctor / status
            Test: all commands return exit code 0 on happy path

PHASE 7   Ollama Compatibility Endpoints
            ollama_compat.py + gateway.py
            Test: Open WebUI connects and chat works

PHASE 8   Plugin System Base
            base.py + plugin loader
            Test: load a mock plugin, verify all 4 hooks are called

PHASE 9   RAG Connector Plugin
            rag_connector/ — requires embedding model + vector DB
            Test: retrieve from a 10-document index, assert recall ≥ 80%

PHASE 10  Context Cache Plugin
            context_cache/
            Test: second request with identical system prompt ≤ 100ms TTFT

PHASE 11  Web Dashboard
            React SPA — all pages functional
            Test: all 8 web dashboard audit checks pass

PHASE 12  Community Profile Registry
            index_client.py + phantom-models/ submodule
            Test: phantom pull fetches a community profile when available

PHASE 13  Updated Installer + Migration Guide
            install.sh extended for full platform
            OLLAMA_MIGRATION.md complete and accurate
```

---

## 17. Differentiation Summary

The single most important output in the entire PHANTOM platform is this terminal response, producible before the user downloads a single model:

```bash
$ phantom plan llama3:70b

  Ollama on this hardware (RTX 4050, 6GB VRAM):
    ✗  llama3:70b  — CANNOT RUN  (requires ~48GB VRAM)
    ✓  llama3:8b   — Maximum model on native hardware

  PHANTOM on this hardware:
    ✓  llama3:70b  — FULLY SUPPORTED
    ✓  Estimated speed:    4.2 tok/sec
    ✓  Context support:    96K tokens  (vs ~4K with 6GB VRAM in Ollama)
    ✓  Simultaneous:       llama3:70b + phi3:3.8b can coexist
    ✓  Ceiling lift:       10× beyond native hardware capacity

  Ready to run?
    phantom pull llama3:70b && phantom run llama3:70b
```

That output is the product. Every line of code in PHANTOM exists to make that output true.

---

## Appendix A — Suggested v1.1 Additions

After shipping v1.0, these are the highest-ROI additions for community growth:

| Feature | Impact | Effort |
|---|---|---|
| Community profile sharing (`phantom push`) | Very High | Medium |
| LoRA adapter support in Phantomfile | High | High |
| ROCm backend (AMD GPU) | High | Very High |
| `phantom benchmark` — contribute to leaderboard | High | Low |
| Prometheus metrics exporter (`/metrics` in Prometheus format) | Medium | Low |
| Grafana dashboard template | Medium | Low |
| MCP tool server integration in ToolRouterPlugin | High | Medium |
| `phantom quantize` — local re-quantization without conversion | Medium | High |
| Mobile API client library (Swift/Kotlin) | Medium | Medium |

---

## Appendix B — Known Architectural Constraints

These are limitations to document clearly and not paper over:

1. **NVIDIA-only (v1):** The entire CUDA stack requires an NVIDIA GPU with Pascal architecture or newer (SM 6.0+). AMD and Apple Silicon are roadmap items.

2. **Linux primary, WSL2 for Windows:** `io_uring` (used by Phantom Pages for NVMe I/O) is Linux-specific. Windows users must use WSL2. Native Windows would require replacing `tokio-uring` with Windows IOCP.

3. **Lossy by default:** Spectral Quantization and Neural Cache KV compression introduce small quality losses. `PHANTOM_PARAM safe_mode on` in a Phantomfile disables all lossy innovations for users who need exact FP16 equivalence.

4. **Calibration is hardware-specific:** A `.phantom` calibration profile generated on an RTX 4090 is not optimal for an RTX 4050. Community-shared profiles are "better than nothing" but models should ideally be re-calibrated on each machine.

5. **Mixtral / MoE models need special handling:** The Adaptive Compute Routing innovation (sparse neuron routing) interacts with Mixture-of-Experts architecture in a non-trivial way. MoE expert routing and PHANTOM's sparsity gate are two different sparsity mechanisms. Mixtral models must disable PHANTOM's sparsity routing (`PHANTOM_PARAM sparsity_routing off`) until a proper MoE-aware implementation is added in v1.1.

---

*PHANTOM — Run the Unreachable.*

*Every byte of VRAM. Every page of RAM. Every block of NVMe.*
*The model that was impossible on your hardware yesterday? It runs today.*
