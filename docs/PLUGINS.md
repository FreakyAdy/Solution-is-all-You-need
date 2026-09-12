# PHANTOM Plugin Middleware System
### Extensible Middleware Architecture for Local LLM Inference

The PHANTOM Plugin System allows developers to intercept and augment the model lifecycle across four execution hooks:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 INCOMING REQUEST                       │
                  └──────────────────────────┬─────────────────────────────┘
                                             │
                                   ┌─────────▼─────────┐
                                   │    pre_request    │ (Auth, prompt sanitization)
                                   └─────────┬─────────┘
                                             │
                                   ┌─────────▼─────────┐
                                   │   pre_generate    │ (RAG injection, tool schemas)
                                   └─────────┬─────────┘
                                             │
                                   ┌─────────▼─────────┐
                                   │     on_token      │ (Token streaming, stop words)
                                   └─────────┬─────────┘
                                             │
                                   ┌─────────▼─────────┐
                                   │   post_generate   │ (Tool call dispatch, caching)
                                   └─────────┬─────────┘
                                             │
                  ┌──────────────────────────▼─────────────────────────────┐
                  │                 COMPLETED RESPONSE                     │
                  └────────────────────────────────────────────────────────┘
```

---

## 1. Plugin Base Protocol

Custom plugins inherit from `BasePlugin` (`phantom.plugins.base`):

```python
from phantom.plugins.base import BasePlugin, GenerationContext

class CustomPlugin(BasePlugin):
    name = "my-plugin"
    version = "1.0.0"

    async def pre_request(self, request_data: dict) -> dict:
        """Inspect or mutate raw API request payload before execution."""
        return request_data

    async def pre_generate(self, prompt: str, context: GenerationContext) -> str:
        """Augment or transform prompt before model prefill."""
        return prompt

    async def on_token(self, token: str, context: GenerationContext) -> Optional[str]:
        """Inspect or alter token during streaming generation."""
        return token

    async def post_generate(self, response: str, context: GenerationContext) -> str:
        """Inspect or mutate generated completion before returning to client."""
        return response
```

---

## 2. Built-in Plugins

### 2.1 RAG Connector (`PLUGIN rag-connector`)
Retrieves relevant documents from a local vector store (ChromaDB / LanceDB / SQLite-VSS) and injects contextual chunks directly into the prompt before generation.

**Configuration (`~/.phantom/plugins/rag/config.toml`):**
```toml
[rag]
db_path = "~/.phantom/rag/default"
top_k = 5
embedding_model = "nomic-embed-text"
max_context_tokens = 2048
similarity_threshold = 0.72
```

### 2.2 Tool Router & MCP Client (`PLUGIN tool-router`)
Enables OpenAI-compatible function calling and Model Context Protocol (MCP) tool routing:
1. **Python Functions**: Native callables decorated with `@phantom_tool`.
2. **HTTP Webhooks**: REST endpoints invoked with JSON payloads.
3. **MCP Tool Servers**: Connect to MCP servers via JSON-RPC 2.0 (`tools/list`, `tools/call`).

**Example: Defining a Local Tool:**
```python
from phantom.plugins.tool_router.plugin import phantom_tool

@phantom_tool
def get_stock_price(symbol: str) -> str:
    """Fetch real-time stock price for a ticker symbol."""
    return f"{symbol.upper()}: $184.20 (+1.4%)"
```

**Example: Connecting an MCP Server:**
```python
from phantom.plugins.tool_router.plugin import ToolRouterPlugin

router = ToolRouterPlugin()
router.register_mcp_server(
    name="filesystem-mcp",
    endpoint="http://localhost:8080/mcp",
    tools=[{"name": "read_file", "description": "Read local file contents"}]
)
```

### 2.3 Context Cache (`PLUGIN context-cache`)
Caches KV state of repeated system prompts. Utilizing PHANTOM's **Neural Cache** autoencoder, cached states are compressed by **8×**, enabling sub-100ms Time-To-First-Token (TTFT) on repeat prompts while consuming negligible VRAM.

**Configuration (`~/.phantom/plugins/context_cache/config.toml`):**
```toml
[context_cache]
max_cached_prefixes = 16
min_prefix_tokens = 128
compression_tier = "spectral_fp8"
ttl_seconds = 3600
```

---

## 3. Registering Plugins in a Phantomfile

You can enable plugins declaratively in any model's `Phantomfile`:

```dockerfile
FROM llama3:70b

SYSTEM """
You are an enterprise research assistant equipped with local file search and calculation tools.
"""

# Enable middleware plugins
PLUGIN rag-connector
PLUGIN tool-router
PLUGIN context-cache

PHANTOM_PARAM sparsity_routing 0.60
PHANTOM_PARAM kv_compression 8.0
```

Build and run:
```bash
phantom create enterprise-assistant -f Phantomfile
phantom run enterprise-assistant
```
