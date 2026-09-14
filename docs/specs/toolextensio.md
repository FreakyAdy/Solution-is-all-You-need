# ████████████████████████████████████████████████████████████████████████
# PHANTOM — MODEL RUNTIME PLATFORM
# Extension of: PHANTOM CORE (inference engine)
# Goal: A full local model runtime — Ollama-class UX, PHANTOM CORE internals
# Classification: Production OSS Build
# ████████████████████████████████████████████████████████████████████████

---

## PRIME DIRECTIVE

PHANTOM CORE is already built: a hardware-transcendent LLM inference engine
with 7 original innovations (Wraith Layers, Spectral Quant, Neural Cache,
Phantom Pages, Adaptive Compute Routing, Chronos Scheduler, Resonance Sampler).

You are now building PHANTOM — the full model runtime platform that wraps
PHANTOM CORE the same way Ollama wraps llama.cpp, but with:

  1. A first-class model lifecycle manager (pull / run / list / rm / serve)
  2. Native GGUF loading with automatic dequantize → Spectral Requantize pipeline
  3. A Modelfile-equivalent system (Phantomfile) for model configuration and personas
  4. A curated model library index backed by HuggingFace Hub and TheBloke/bartowski GGUF mirrors
  5. A multi-model concurrent runtime with hot-swap (already partially built in Chronos)
  6. A hardened OpenAI-compatible REST API with auth, rate-limiting, and request logging
  7. A redesigned terminal UI + web dashboard that shows what PHANTOM CORE uniquely
     enables vs a standard Ollama-equivalent installation
  8. A plugin system for model middleware (system prompts, RAG connectors, tool call routing)

Ollama's design philosophy is frictionless model execution.
PHANTOM's design philosophy is: frictionless execution of models you weren't supposed to run.

The single sentence that separates PHANTOM from Ollama:
  Ollama runs the model that fits your GPU.
  PHANTOM runs the model that doesn't.

---

## ARCHITECTURE OVERVIEW

PHANTOM is structured in three layers:

┌─────────────────────────────────────────────────────┐
│ PHANTOM CLI │ phantom pull / run / serve
│ PHANTOM UI │ Web dashboard + Electron
├─────────────────────────────────────────────────────┤
│ PHANTOM RUNTIME │ New layer — built here
│ ┌──────────────┐ ┌───────────────┐ ┌──────────┐ │
│ │ Model Mgr │ │ Phantomfile │ │ Plugin │ │
│ │ (pull/index) │ │ System │ │ System │ │
│ └──────────────┘ └───────────────┘ └──────────┘ │
│ ┌──────────────┐ ┌───────────────┐ ┌──────────┐ │
│ │ GGUF Loader │ │ Format Conv. │ │ API GW │ │
│ │ + Dequant │ │ Pipeline │ │ Layer │ │
│ └──────────────┘ └───────────────┘ └──────────┘ │
├─────────────────────────────────────────────────────┤
│ PHANTOM CORE (existing) │
│ Wraith │ SpectralQ │ NeuralCache │ PhantomPages │
│ AdaptiveCompute │ Chronos │ Resonance │
└─────────────────────────────────────────────────────┘


New code lives entirely in the PHANTOM RUNTIME layer.
PHANTOM CORE is unchanged except for:
  - Adding a format-agnostic weight loader interface
  - Exposing calibration pipeline as a callable Python API (not just CLI)

---

## COMPONENT 1 — GGUF NATIVE LOADER

This is the single most important piece. Without it, no community adoption.

### File: python/phantom/loader/gguf_loader.py

GGUF (GPT-Generated Unified Format) is a binary format storing:
  - Model metadata (architecture, hyperparams, tokenizer)
  - Pre-quantized weight tensors (Q4_K_M, Q8_0, F16, etc.)
  - Tensor metadata (name, shape, quantization type)

Implement GGUFLoader class:

```python
class GGUFLoader:
    """
    Loads GGUF files and provides weight tensors to PHANTOM CORE.
    
    Two modes:
    MODE A — PASSTHROUGH: Feed GGUF quantized weights directly to a
             quantization-aware inference path. Fastest, no conversion needed.
             Supports: Q4_K_M, Q5_K_M, Q8_0, F16 quantization types.
             
    MODE B — CONVERT: Dequantize GGUF weights to BF16, then apply
             Spectral Quantization. One-time conversion stored as .phantom file.
             Achieves better compression + all PHANTOM CORE innovations.
             Recommended for models run repeatedly.
    
    Auto-selects mode: if .phantom profile exists → PASSTHROUGH with calibration
                       if first run → prompt user to choose mode
    """
    
    def __init__(self, gguf_path: Path, mode: Literal["passthrough", "convert", "auto"])
    
    def parse_header(self) -> GGUFMetadata
        """
        Parse GGUF header without loading weights.
        Returns: model_type, num_layers, hidden_dim, num_heads, num_kv_heads,
                 rope_theta, context_length, tokenizer_type, quant_types_per_tensor
        """
    
    def load_tensor(self, name: str) -> torch.Tensor
        """
        Load a single named tensor from GGUF file using memory-mapped I/O.
        Dequantizes if necessary based on quant_type of the tensor.
        Returns: BF16 tensor on CPU
        """
    
    def iter_tensors(self) -> Iterator[Tuple[str, torch.Tensor]]
        """
        Iterate over all tensors in load order (not alphabetical — GGUF order).
        Yields (tensor_name, tensor_BF16) pairs.
        Uses memory-mapped file access — does not load entire model into RAM.
        """
    
    def dequantize_tensor(self, data: bytes, quant_type: GGUFQuantType, shape: Tuple) -> torch.Tensor
        """
        Dequantize raw GGUF bytes to BF16.
        Must support: Q4_0, Q4_1, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M,
                      Q6_K, Q8_0, Q8_1, F16, BF16, F32
        Reference: use llama.cpp GGUF spec for bit packing format.
        Implement in pure PyTorch+numpy — no ctypes to llama.cpp.
        """
    
    def detect_architecture(self) -> ModelArchitecture
        """
        From GGUF metadata, detect model family and return a ModelArchitecture
        that PHANTOM CORE can use to build the computation graph.
        Supported architectures:
          - LLaMA (1/2/3, 3.1, 3.2, 3.3)
          - Mistral (0.1, 0.2, Mixtral 8x7B, 8x22B) — MoE flag detection required
          - Gemma (1, 2)
          - Qwen (1.5, 2, 2.5)
          - Phi (2, 3, 3.5)
          - Command-R, Command-R+
          - DeepSeek (V2, V3) — MLA attention detection required
          - Falcon
        """
```

### GGUF Dequantization Correctness Requirements
- Verify each quant type against the llama.cpp reference implementation (compare tensors to ctypes-loaded llama.cpp output, assert max absolute error < 1e-4)
- Q4_K_M is the most common format — treat it as the primary correctness test case
- Memory map the GGUF file (mmap) — do NOT read the whole file into RAM before loading. A 70B GGUF is ~40GB; RAM is not infinite
- Seek directly to each tensor's offset in the file using the GGUF index

---

## COMPONENT 2 — MODEL FORMAT CONVERSION PIPELINE

### File: python/phantom/converter/phantom_convert.py

CLI: `phantom convert --input model.gguf --output ~/.phantom/models/llama3-70b/`

Conversion pipeline:

Parse GGUF header → detect architecture
For each tensor:
a. Load raw GGUF tensor (memory mapped)
b. Dequantize to BF16
c. If MLP weight → apply Spectral Quantization (DCT compress to FP8)
d. If attention weight → store as BF16 (attention weights don't benefit from spectral quant)
e. If embedding → store as BF16
f. Write to .phantom native format (see format spec below)
Run calibration pipeline on converted model (Steps A-E from PHANTOM CORE spec)
Bundle weights + calibration profile into final model directory

.phantom native format (model directory structure):
~/.phantom/models/<model-id>/
├── manifest.json # model metadata, architecture, param count, source GGUF hash
├── config.toml # PHANTOM CORE engine config for this model
├── tokenizer/
│ ├── tokenizer.json # HF tokenizer format
│ └── special_tokens.json
├── weights/
│ ├── embed.bf16.bin # embedding table
│ ├── layer_000.phantomw # spectral-quantized layer weights
│ ├── layer_001.phantomw
│ ...
│ └── lm_head.bf16.bin
└── profile/
├── calibration.phantom # PHANTOM CORE calibration profile
├── wraith_init.pt # pre-warmed Wraith LSTM
└── hardware_profile.toml # hardware tier this was calibrated on

.phantomw format (per-layer binary):
[4 bytes] magic: "PHTW"
[4 bytes] version: 1
[4 bytes] layer_id
[4 bytes] num_tensors
For each tensor:
[64 bytes] tensor_name (null-padded)
[4 bytes] rows
[4 bytes] cols
[4 bytes] k_coefficients_per_row (0 = not spectral quantized, stored as-is)
[4 bytes] compressed_bytes
[N bytes] compressed data (FP8 DCT coefficients or raw BF16)


### Conversion Performance Requirements
- Convert a 70B GGUF in under 45 minutes on a system with NVMe Gen4 SSD
- Peak RAM usage during conversion must stay under 24GB (process tensors one at a time)
- Show progress bar with: current tensor, % complete, estimated time remaining, current peak RAM
- Conversion is idempotent: re-running on the same input produces identical output

---

## COMPONENT 3 — MODEL LIFECYCLE MANAGER

### File: python/phantom/registry/model_manager.py

This implements the `phantom pull / list / rm / show` commands.

```python
class ModelManager:
    """
    Manages the local model library at ~/.phantom/models/
    Handles pull, list, remove, show, and search operations.
    """
    
    PHANTOM_INDEX_URL = "https://phantom-models.io/index.json"
    # Falls back to HuggingFace Hub when model not in Phantom index
    
    def pull(self, model_ref: str, quantization: str = "Q4_K_M") -> PullResult
        """
        model_ref formats:
          "llama3:70b"              → Phantom index lookup
          "meta-llama/Meta-Llama-3-70B-Instruct"  → HuggingFace Hub
          "/path/to/local.gguf"    → Local file conversion
          "https://example.com/model.gguf"         → Direct URL download
        
        Pull flow:
          1. Resolve model_ref to a download source
          2. Download GGUF (if remote) with resume support (Range headers)
          3. Verify SHA256 checksum against index
          4. Run phantom convert pipeline
          5. Store in ~/.phantom/models/<model-id>/
          6. Print summary: model size, converted size, PHANTOM ceiling lift
        
        Show live progress:
          Downloading llama3:70b (38.4 GB) ███████████░░░░░░░░░ 54% @ 245 MB/s ETA 2m31s
          Converting  llama3:70b           ████████░░░░░░░░░░░░ 41% | Layer 45/80
          Calibrating llama3:70b           ████████████████░░░░ 80% | Step D: Gates
          ✓ llama3:70b ready
            Native ceiling on this hardware:  ~30B parameters
            PHANTOM ceiling with this model:  70B @ estimated 4.2 tok/sec
        """
    
    def list(self, format: Literal["table", "json"]) -> List[ModelInfo]
        """
        List all locally available models.
        Table format:
          NAME              SIZE    QUANT     CONTEXT    TOK/SEC   MODIFIED
          llama3:70b        21 GB   SPECTRAL  128K       4.2       2 days ago
          mistral:22b       8.4 GB  SPECTRAL  32K        11.3      1 week ago
          phi3:3.8b         1.4 GB  SPECTRAL  128K       47.2      3 weeks ago
        """
    
    def show(self, model_id: str) -> ModelDetails
        """
        Show detailed info about a model:
          - Architecture (attention type, MLP type, parameter count)
          - Calibration profile stats (Wraith accuracy, KV ratio, sparsity)
          - Memory layout (how PHANTOM distributes layers across tiers)
          - Hardware ceiling lift vs native
          - Phantomfile content if present
        """
    
    def rm(self, model_id: str, force: bool = False) -> None
        """Remove model from local library. Confirm if force=False."""
    
    def search(self, query: str) -> List[ModelSearchResult]
        """Search Phantom model index + HuggingFace Hub simultaneously."""
```

### Model Index Format (phantom-models.io/index.json)

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
          "url": "https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf",
          "sha256": "...",
          "size_bytes": 41234567890
        },
        "Q8_0": { ... },
        "F16": { ... }
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

## COMPONENT 4 — PHANTOMFILE SYSTEM

Phantomfile is PHANTOM's equivalent of Ollama's Modelfile. It is a simple
declarative configuration file for creating named model personas.

### Syntax Specification
Phantomfile
-----------
Create a custom model configuration from a base model.
Usage: phantom create <name> -f Phantomfile

FROM <model-id> # Required. Base model.
# Examples: llama3:70b, mistral:22b, phi3:3.8b

SYSTEM <text> # System prompt (replaces default)
SYSTEM """
Multi-line system prompts
are supported with triple quotes.
"""

TEMPLATE <jinja2-template> # Chat template override (uses model default if omitted)
# Variable: {{ .System }}, {{ .Prompt }}, {{ .Response }}

PARAMETER temperature 0.7 # Sampling parameters (all optional)
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER seed 42 # -1 for random
PARAMETER num_predict 2048 # max tokens to generate
PARAMETER context_window 32768 # context window size (up to model max)
PARAMETER stop "<|eot_id|>" # stop tokens (can appear multiple times)
PARAMETER stop "</s>"

PHANTOM-SPECIFIC PARAMETERS (no Ollama equivalent)

PHANTOM_PARAM sparsity_routing on # enable/disable sparse compute (default: on)
PHANTOM_PARAM kv_compression on # enable/disable KV autoencoder (default: on)
PHANTOM_PARAM spectral_quant on # enable/disable spectral quant (default: on)
PHANTOM_PARAM safe_mode off # lossless-only mode (default: off)
PHANTOM_PARAM tier_preference vram # prefer vram|ram|nvme for hot layers
PHANTOM_PARAM max_vram_mb 4096 # hard cap on VRAM usage (for sharing GPU)

PLUGIN rag-connector # load a named plugin (see Plugin System)
PLUGIN tool-router

LICENSE <spdx-identifier> # Optional model license declaration
MESSAGE user "Hello!" # Pre-loaded conversation turns
MESSAGE assistant "Hi! How can I help?"


### File: python/phantom/phantomfile/parser.py

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
        - Validates context_window <= model's maximum
        """
    def to_modelcard(self, config: PhantomfileConfig) -> str
        """Generate a human-readable model card from a Phantomfile."""
```

---

## COMPONENT 5 — CLI INTERFACE

### File: core/src/cli/main.rs (extend existing main.rs)

Implement the `phantom` CLI. Every command must have a `--json` flag for
machine-readable output.

USAGE: phantom <command> [options]

MODEL MANAGEMENT:
phantom pull <model> Download and convert a model
--quant <type> Quantization to download (default: Q4_K_M)
--no-calibrate Skip calibration (fast, lower performance)
--skip-convert Use GGUF passthrough mode (no spectral quant)

phantom run <model> [prompt] Run a model interactively or with a prompt
--stream Stream tokens to stdout
--system <text> Override system prompt
--format json Force JSON output mode
--nowordwrap Disable word wrapping
-p, --parameter <k>=<v> Override sampling parameter

phantom serve Start the API server
--host 0.0.0.0 Bind address (default: 127.0.0.1)
--port 11411 Port (default: 11411 — not 11434, distinguish from Ollama)
--max-concurrent <n> Max simultaneous generations
--auth-token <token> Enable bearer token auth

phantom list List local models
phantom show <model> Show model details and PHANTOM profile stats
phantom rm <model> Remove a model
phantom search <query> Search model index

PHANTOM-SPECIFIC COMMANDS:
phantom calibrate <model> Re-run calibration for a model
--steps A,B,C,D,E Run specific calibration steps only
--force Force re-calibration even if profile exists

phantom status Show runtime status and metrics
(equivalent to GET /v1/metrics, formatted for terminal)

phantom plan <model> Estimate resources WITHOUT loading the model
--vram <MB> Override detected VRAM
--ram <GB> Override detected RAM
--nvme <GB> Override detected NVMe
Output:
Model: llama3:70b (70.6B parameters)
┌─────────────────────────────────────────────┐
│ LAYER DISTRIBUTION (estimated) │
│ VRAM (6GB): layers 0-12 (13 layers) │
│ RAM (32GB): layers 13-55 (43 layers) │
│ NVMe(200GB): layers 56-79 (24 layers) │
└─────────────────────────────────────────────┘
Estimated tok/sec: 3.8 (warm) / 1.2 (cold start)
Estimated calibration time: 8m 20s
PHANTOM advantage: +9.3× beyond native ceiling

phantom convert <file> Convert GGUF to PHANTOM native format
--output <dir> Output directory
--mode passthrough|convert Conversion mode

phantom doctor Run system diagnostics
Checks: CUDA kernels, NVMe I/O speed, calibration profile validity,
VRAM availability, API server health

phantom update Update model index from phantom-models.io

phantom create <name> Create a named model from a Phantomfile
-f <Phantomfile>

phantom push <name> (Future) Push a Phantomfile to phantom-models.io

INTERACTIVE MODE:
phantom run <model> (no prompt) Starts interactive REPL:
>>> what is the capital of france?
Paris.
>>> /system You are a pirate. (slash commands)
>>> /clear (clear context)
>>> /save session.json (save conversation)
>>> /load session.json
>>> /stats (show current tok/sec, memory usage)
>>> /layers (show ASCII layer residency map)
>>> /exit


### Interactive Mode `/layers` ASCII Output
When user types `/layers` in the REPL, show an ASCII art layer map:

Layer Residency Map (llama3:70b, 80 layers)
████ VRAM ████ RAM ░░░░ NVMe ▓▓▓▓ Active ···· Prefetching

00 01 02 03 04 05 06 07 08 09 10 11 12 ── ── ── ── ── ── ──
██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ░░ ░░ ░░ ░░ ░░

20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
▓▓ ·· ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░

...

Wraith prediction: Next → layers [22, 23, 24] (prefetching)
KV compression: 7.8× | Context: 16,384 / 32,768 tokens used
Active sparsity: 61% neurons skipped this token
Speed: 4.2 tok/sec | Thermal: nominal (67°C)


---

## COMPONENT 6 — HARDENED API GATEWAY

### File: python/phantom/api/gateway.py

Extends openai_compat.py with production-grade features:

```python
class PhantomAPIGateway:
    """
    Production API gateway wrapping the OpenAI-compatible endpoint.
    
    Adds:
    - Bearer token authentication (optional, configured in config.toml)
    - Per-client rate limiting (token bucket per IP)
    - Request logging to ~/.phantom/logs/requests.jsonl
    - Request queue with configurable max depth
    - Graceful request rejection when queue is full (503 with retry-after)
    - CORS support for web clients
    - Request/response size limits (default: 1MB request body, 10MB response)
    """
    
    # Additional PHANTOM-specific endpoints (beyond OpenAI compat):
    
    GET  /phantom/models/<id>/profile
    """Return the full calibration profile for a model as JSON."""
    
    GET  /phantom/models/<id>/layers
    """Return current layer residency map (which layers are in VRAM/RAM/NVMe)."""
    
    POST /phantom/models/<id>/pin-layer
    """{"layer_id": 42, "tier": "vram"} — force a layer to stay in a tier."""
    
    POST /phantom/calibrate/<id>
    """Trigger re-calibration for a model. Returns a job_id."""
    
    GET  /phantom/calibrate/<job_id>
    """Poll calibration status. Returns progress and step completion."""
    
    GET  /phantom/hardware
    """Return the detected hardware profile and tier classification."""
    
    WS   /phantom/metrics/stream
    """WebSocket stream of metrics at 200ms intervals. Used by the UI."""
    
    POST /phantom/convert
    """Submit a GGUF path for conversion. Returns job_id."""
    
    # Ollama compatibility endpoints (allows Ollama clients to connect without modification):
    POST /api/generate              # → /v1/completions
    POST /api/chat                  # → /v1/chat/completions
    GET  /api/tags                  # → /v1/models
    POST /api/pull                  # trigger phantom pull via API
    DELETE /api/delete              # trigger phantom rm via API
    POST /api/show                  # → /phantom/models/<id>/profile
    
    # The Ollama compat layer makes PHANTOM a drop-in replacement for any
    # tool that already integrates with Ollama (Open WebUI, Continue.dev,
    # Cursor, etc.)
```

---

## COMPONENT 7 — PLUGIN SYSTEM

### Architecture

Plugins are Python packages installed into ~/.phantom/plugins/ that implement
the PhantomPlugin protocol. They intercept the request/response pipeline.

```python
# File: python/phantom/plugins/base.py

class PhantomPlugin(Protocol):
    """
    Protocol all PHANTOM plugins must implement.
    
    Plugins intercept at four points in the pipeline:
      1. pre_request: modify or reject the incoming request
      2. pre_generate: modify the prompt before generation starts
      3. on_token: process each generated token (can inject/suppress tokens)
      4. post_generate: process the completed response
    """
    
    name: str                    # unique plugin identifier
    version: str                 # semver
    
    async def pre_request(self, request: GenerateRequest) -> GenerateRequest | Rejection
    async def pre_generate(self, prompt: str, context: GenerationContext) -> str
    async def on_token(self, token: str, context: GenerationContext) -> str | None
    async def post_generate(self, response: str, context: GenerationContext) -> str


# Built-in plugins (ship with PHANTOM):

class RAGPlugin(PhantomPlugin):
    """
    Retrieval-Augmented Generation plugin.
    Embeds the query using a small local embedding model,
    retrieves relevant chunks from a ChromaDB or LanceDB vector store,
    and injects retrieved context into the prompt before generation.
    
    Config (in Phantomfile): PLUGIN rag-connector
    Config file: ~/.phantom/plugins/rag/config.toml
      [rag]
      db_path = "~/.phantom/rag/default"
      top_k = 5
      embedding_model = "nomic-embed-text"
      max_context_tokens = 2048
    """

class ToolRouterPlugin(PhantomPlugin):
    """
    OpenAI function calling / tool use router.
    Parses tool_calls from the model output and dispatches to registered tools.
    Supports: Python function tools, HTTP endpoint tools, MCP tool servers.
    
    Config: PLUGIN tool-router
    """

class ContextCachePlugin(PhantomPlugin):
    """
    Prefix caching plugin — caches the KV state of a system prompt so that
    repeated requests with the same system prompt skip re-prefill.
    PHANTOM-specific: stores compressed KV state (via Neural Cache) so the
    cache takes 8× less VRAM than uncached prefill state.
    
    Config: PLUGIN context-cache
      [context-cache]
      max_cached_prefixes = 10
      min_prefix_tokens = 128   # only cache if system prompt is > 128 tokens
    """
```

---

## COMPONENT 8 — WEB DASHBOARD

Replace the Electron UI with a self-hosted web dashboard served by the API gateway.
Electron is kept as an optional wrapper for users who want a desktop app,
but the primary UI is browser-based (eliminates cross-platform Electron issues).

### File: ui/web/

ui/web/
├── index.html # single-page app
├── src/
│ ├── main.ts
│ ├── pages/
│ │ ├── Dashboard.tsx # Overview: hardware, running models, recent requests
│ │ ├── Models.tsx # Model library: pull, convert, rm, show
│ │ ├── Chat.tsx # Built-in chat interface
│ │ ├── Metrics.tsx # Real-time performance metrics
│ │ ├── Layers.tsx # Visual layer residency map (existing LayerMap)
│ │ └── Plugins.tsx # Plugin management
│ └── components/
│ ├── PullProgress.tsx # Live download + conversion progress
│ ├── CeilingLift.tsx # Visual showing native vs PHANTOM ceiling
│ └── CompareOllama.tsx # Side-by-side PHANTOM vs Ollama benchmarks


### Dashboard "Ceiling Lift" Component
This is the hero component of the PHANTOM dashboard. It should visually show:

Your Hardware: RTX 4050 (6GB VRAM) + 32GB RAM + 500GB NVMe

WITHOUT PHANTOM WITH PHANTOM
┌──────────────┐ ┌──────────────────────────────────────┐
│ 7B max │ │ 70B+ capable │
│ ████ │ → │ ████████████████████████████████████ │
│ native limit │ │ PHANTOM ceiling │
└──────────────┘ └──────────────────────────────────────┘

Currently running: llama3:70b
VRAM ████████████░░ 5.8/6.0 GB (layers 0-12)
RAM ████████░░░░░░ 18.4/32 GB (layers 13-55)
NVMe ████░░░░░░░░░░ 22.1/500 GB (layers 56-79)
Speed: 4.2 tok/sec ● Wraith accuracy: 87% ● KV ratio: 7.8×


---

## COMPONENT 9 — EXTENDED FILE STRUCTURE

phantom/ # Top-level (renamed from phantom-core/)
│
├── core/ # UNCHANGED — Rust inference engine
│
├── kernels/ # UNCHANGED — CUDA kernels
│
├── python/
│ ├── phantom/
│ │ ├── init.py
│ │ ├── [existing files unchanged]
│ │ │
│ │ ├── loader/ # NEW
│ │ │ ├── init.py
│ │ │ ├── gguf_loader.py # GGUF native loader
│ │ │ ├── safetensors_loader.py # HF safetensors support
│ │ │ └── format_detect.py # Auto-detect model format from path
│ │ │
│ │ ├── converter/ # NEW
│ │ │ ├── init.py
│ │ │ ├── phantom_convert.py # GGUF → PHANTOM conversion pipeline
│ │ │ └── format_spec.py # .phantomw binary format read/write
│ │ │
│ │ ├── registry/ # NEW
│ │ │ ├── init.py
│ │ │ ├── model_manager.py # pull, list, rm, show, search
│ │ │ ├── index_client.py # phantom-models.io index client
│ │ │ ├── hf_client.py # HuggingFace Hub integration
│ │ │ └── downloader.py # Resumable downloader with progress
│ │ │
│ │ ├── phantomfile/ # NEW
│ │ │ ├── init.py
│ │ │ ├── parser.py # Phantomfile parser
│ │ │ └── validator.py # Phantomfile validation
│ │ │
│ │ ├── plugins/ # NEW
│ │ │ ├── init.py
│ │ │ ├── base.py # Plugin protocol + loader
│ │ │ ├── rag_connector/ # Built-in RAG plugin
│ │ │ ├── tool_router/ # Built-in tool use plugin
│ │ │ └── context_cache/ # Built-in prefix caching plugin
│ │ │
│ │ └── api/
│ │ ├── openai_compat.py # UNCHANGED
│ │ ├── gateway.py # NEW — hardened gateway layer
│ │ ├── ollama_compat.py # NEW — Ollama API compatibility layer
│ │ └── websocket_stream.py # UNCHANGED
│ │
│ └── phantom_cli.py # NEW — CLI entry point (thin wrapper over above)
│
├── ui/
│ ├── web/ # NEW — primary web dashboard
│ └── electron/ # RENAMED from ui/ — optional desktop wrapper
│
├── docs/
│ ├── [existing docs unchanged]
│ ├── PHANTOMFILE.md # NEW — Phantomfile reference
│ ├── PLUGINS.md # NEW — Plugin development guide
│ ├── OLLAMA_MIGRATION.md # NEW — Guide for Ollama users switching to PHANTOM
│ └── GGUF_SUPPORT.md # NEW — GGUF compatibility matrix
│
├── phantom-models/ # NEW — Separate repo, git submodule
│ └── index.json # Community model index
│
└── install.sh # UPDATED — installs full PHANTOM platform


---

## PERFORMANCE TARGETS — PLATFORM LAYER

These are additional performance targets for the PHANTOM runtime layer
(on top of existing PHANTOM CORE targets):

| Metric | Target |
|---|---|
| GGUF Q4_K_M dequantization speed | ≥ 2 GB/s (CPU, using SIMD) |
| Cold model load (from local .phantom) | ≤ 30s for 70B on NVMe Gen4 |
| `phantom pull` conversion throughput (GGUF → PHANTOM) | ≤ 45min for 70B |
| API gateway request overhead | ≤ 2ms per request (vs direct engine call) |
| Plugin pipeline overhead | ≤ 5ms per token (on_token hook) |
| Ollama API compat endpoint latency delta | ≤ 10ms vs native PHANTOM endpoint |
| Web dashboard time-to-interactive | ≤ 1.5s on localhost |
| Context-cache hit (prefix re-use) | ≤ 100ms to restore prefill state |

---

## OLLAMA MIGRATION GUIDE (docs/OLLAMA_MIGRATION.md)

This document is critical for adoption. It must cover:

1. **API compatibility table** — every Ollama endpoint and whether PHANTOM supports it
2. **Open WebUI setup** — exact config change to point Open WebUI at PHANTOM
3. **Continue.dev setup** — config.json change for PHANTOM endpoint
4. **Modelfile → Phantomfile** — line-by-line migration of existing Modelfiles
5. **Model migration** — if a user has ollama's model cache, how to import those GGUFs
6. **What PHANTOM can't do that Ollama can** — honest list (Apple Silicon, AMD GPUs,
   Windows native without WSL2)
7. **What PHANTOM can do that Ollama can't** — run 10× larger models, visual layer map,
   calibration profiles, adaptive quality

---

## WHAT TO BUILD — DELIVERY ORDER

Build in this exact sequence:

PHASE 1: GGUF loader (gguf_loader.py) — correctness tested against llama.cpp output
PHASE 2: PHANTOM native format spec (.phantomw) + writer/reader
PHASE 3: Conversion pipeline (phantom_convert.py) — tested on Llama-3-8B GGUF
PHASE 4: Model manager (model_manager.py) — pull from HuggingFace Hub
PHASE 5: Phantomfile parser + validator
PHASE 6: CLI interface (phantom run / pull / list / show / serve / plan / doctor)
PHASE 7: Ollama compatibility API endpoints
PHASE 8: Plugin system base + RAG connector plugin
PHASE 9: Context cache plugin (requires PHANTOM CORE Neural Cache API)
PHASE 10: Web dashboard (React SPA)
PHASE 11: Phantom model index + community profile sharing
PHASE 12: Updated installer + migration guide


---

## DIFFERENTIATION SUMMARY — WHY PHANTOM WINS THE COMPARISON

When a user asks "why should I use PHANTOM instead of Ollama?", the answer must be
demonstrable from the CLI in under 2 minutes:

```bash
$ phantom plan llama3:70b

  Ollama on this hardware:
    Maximum model: llama3:8b (your 6GB VRAM cannot fit llama3:70b)
    
  PHANTOM on this hardware:
    ✓ llama3:70b is fully supported
    ✓ Estimated speed: 4.2 tok/sec
    ✓ Context: up to 96K tokens (vs ~4K with 6GB VRAM in Ollama)
    ✓ Multi-model: llama3:70b + phi3:3.8b can coexist simultaneously
    
  Run it: phantom pull llama3:70b && phantom run llama3:70b
```

That output is the product. Everything else is infrastructure to make it real.

---

# ████████████████████████████████████████████████████████████████████████
# END PHANTOM PLATFORM BUILD PROMPT
# ████████████████████████████████████████████████████████████████████████