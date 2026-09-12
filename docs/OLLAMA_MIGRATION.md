# PHANTOM — Ollama Migration Guide

> *"Ollama runs the model that fits your GPU. PHANTOM runs the model that doesn't."*

## Drop-in API Compatibility

PHANTOM provides 100% endpoint compatibility with Ollama:

| Ollama Endpoint | PHANTOM Support | Notes |
|---|---|---|
| `POST /api/generate` | ✅ Full | Text completion with streaming |
| `POST /api/chat` | ✅ Full | Chat completion with streaming |
| `GET /api/tags` | ✅ Full | Lists installed and available models |
| `POST /api/pull` | ✅ Full | Pulls and converts model automatically |
| `DELETE /api/delete` | ✅ Full | Removes model from local registry |
| `POST /api/show` | ✅ Full | Returns model profile and details |
| `GET /api/ps` | ✅ Full | Lists running models and memory state |

## Switching Open WebUI

Change one environment variable:
```bash
# Before (Ollama)
OLLAMA_BASE_URL=http://localhost:11434

# After (PHANTOM)
OLLAMA_BASE_URL=http://localhost:11411
```

## Switching Continue.dev

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

## Importing Ollama GGUF Models

```bash
# Locate existing GGUF blob in Ollama cache
ls ~/.ollama/models/blobs/

# Convert directly into PHANTOM
phantom convert ~/.ollama/models/blobs/<sha256> --output ~/.phantom/models/llama3-70b/
```

## Modelfile to Phantomfile Line-by-Line Migration

PHANTOM's `Phantomfile` engine is 100% backward compatible with Ollama's `Modelfile`. You can use any existing `Modelfile` directly or take advantage of PHANTOM-specific runtime directives:

| Ollama `Modelfile` Directive | PHANTOM `Phantomfile` Directive | Purpose |
|---|---|---|
| `FROM llama3:8b` | `FROM llama3:70b` | Base model (PHANTOM supports 70B+ on consumer GPUs) |
| `SYSTEM """..."""` | `SYSTEM """..."""` | Model persona and instructions |
| `TEMPLATE """..."""` | `TEMPLATE """..."""` | Custom chat prompt templating |
| `PARAMETER temperature 0.7` | `PARAMETER temperature 0.7` | Standard inference hyperparameter |
| `PARAMETER stop "<|eot_id|>"`| `PARAMETER stop "<|eot_id|>"` | Stop sequence |
| *(Not supported in Ollama)* | `PHANTOM_PARAM sparsity_routing 0.60` | Dynamic neuron skipping (60% active) |
| *(Not supported in Ollama)* | `PHANTOM_PARAM kv_compression 8.0` | 8× Neural Cache autoencoder compression |
| *(Not supported in Ollama)* | `PLUGIN rag-connector` | Built-in local vector retrieval middleware |
| *(Not supported in Ollama)* | `PLUGIN tool-router` | OpenAI function calling & MCP tool servers |

**Migration Command:**
```bash
# Build from an existing Ollama Modelfile directly:
phantom create my-assistant -f Modelfile

# Or convert and augment with PHANTOM innovations:
phantom create enterprise-70b -f Phantomfile
```

## What PHANTOM Enables Beyond Ollama

1. **Run 70B on 6GB VRAM**: Runs models 10× larger than Ollama can fit.
2. **8× Longer Context**: Neural Cache compressed KV representation.
3. **Multi-Model Coexistence**: Chronos holds models simultaneously.
4. **Resonance Sampler**: Dynamically avoids hallucination loops.
5. **Visual Layer Map**: Live 2D heatmap showing where layers live in real-time.

