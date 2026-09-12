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

## What PHANTOM Enables Beyond Ollama

1. **Run 70B on 6GB VRAM**: Runs models 10× larger than Ollama can fit.
2. **8× Longer Context**: Neural Cache compressed KV representation.
3. **Multi-Model Coexistence**: Chronos holds models simultaneously.
4. **Resonance Sampler**: Dynamically avoids hallucination loops.
5. **Visual Layer Map**: Live 2D heatmap showing where layers live in real-time.
