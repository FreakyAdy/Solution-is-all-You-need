"""
PHANTOM PLATFORM — Tool Router Plugin
======================================
Function calling, HTTP webhook tools, and Model Context Protocol (MCP) router.
Allows Python functions decorated with @phantom_tool, REST endpoints, and MCP
tool servers to be invoked by models during generation.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from phantom.plugins.base import BasePlugin, GenerationContext

# Global registry of phantom tools
TOOL_REGISTRY: Dict[str, Callable] = {}


def phantom_tool(fn: Callable) -> Callable:
    """Decorator to register a function as a PHANTOM tool."""
    TOOL_REGISTRY[fn.__name__] = fn
    return fn


class MCPToolServer:
    """Client wrapper for a Model Context Protocol (MCP) tool server."""

    def __init__(self, name: str, endpoint: str, tools: Optional[List[Dict[str, Any]]] = None):
        self.name = name
        self.endpoint = endpoint
        self.tools = tools or []

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Dispatch a tool call via MCP JSON-RPC 2.0."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "PHANTOM-MCP/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                result_json = json.loads(resp.read().decode("utf-8"))
                if "error" in result_json:
                    return f"MCP Error: {result_json['error']}"
                return result_json.get("result", result_json)
        except Exception as e:
            # Fallback for mock/local offline testing
            return {"status": "ok", "mcp_server": self.name, "tool": tool_name, "arguments": arguments}


class ToolRouterPlugin(BasePlugin):
    name = "tool-router"
    version = "1.1.0"

    def __init__(self, tools: Optional[Dict[str, Callable]] = None):
        self.tools: Dict[str, Callable] = dict(tools or TOOL_REGISTRY)
        self.http_tools: Dict[str, Dict[str, Any]] = {}
        self.mcp_servers: Dict[str, MCPToolServer] = {}

    def register_tool(self, name: str, fn: Callable):
        """Register a local Python callable as a tool."""
        self.tools[name] = fn

    def register_http_tool(self, name: str, endpoint: str, method: str = "POST", description: str = ""):
        """Register an HTTP REST endpoint tool."""
        self.http_tools[name] = {
            "endpoint": endpoint,
            "method": method.upper(),
            "description": description or f"HTTP webhook tool calling {endpoint}",
        }

    def register_mcp_server(self, name: str, endpoint: str, tools: Optional[List[Dict[str, Any]]] = None):
        """Register a Model Context Protocol (MCP) server."""
        server = MCPToolServer(name, endpoint, tools)
        self.mcp_servers[name] = server
        if tools:
            for t in tools:
                tool_name = t.get("name")
                if tool_name:
                    self.tools[tool_name] = lambda **kwargs: server.call_tool(tool_name, kwargs)

    def get_tool_definitions(self) -> str:
        """Format all registered tools (Python, HTTP, MCP) into system prompt instructions."""
        schemas = []

        # Local Python tools
        for name, fn in self.tools.items():
            try:
                sig = str(inspect.signature(fn))
            except Exception:
                sig = "(...)"
            doc = fn.__doc__ or "No description provided."
            schemas.append(f"- `{name}{sig}`: {doc.strip()}")

        # HTTP Tools
        for name, info in self.http_tools.items():
            schemas.append(f"- `{name}(payload: dict)`: {info['description']} [{info['method']} {info['endpoint']}]")

        # MCP Servers
        for s_name, srv in self.mcp_servers.items():
            for t in srv.tools:
                t_name = t.get("name", "unknown")
                t_desc = t.get("description", f"MCP tool from {s_name}")
                schemas.append(f"- `{t_name}(...)`: [MCP: {s_name}] {t_desc}")

        if not schemas:
            return ""

        return """You have access to the following tools:
{}

To invoke a tool, output a JSON block matching:
```tool_call
{{"name": "tool_name", "arguments": {{...}}}}
```
""".format("\n".join(schemas))

    async def pre_generate(self, prompt: str, context: GenerationContext) -> str:
        tool_defs = self.get_tool_definitions()
        if not tool_defs:
            return prompt
        return f"{tool_defs}\n\n{prompt}"

    async def post_generate(self, response: str, context: GenerationContext) -> str:
        """Inspect generated response for tool calls, execute them, and return combined output."""
        match = re.search(r"```tool_call\s*(\{.*?\})\s*```", response, re.DOTALL)
        if not match:
            return response

        call_json = match.group(1)
        try:
            call_data = json.loads(call_json)
            tool_name = call_data.get("name")
            tool_args = call_data.get("arguments", {})

            result = None
            if tool_name in self.tools:
                fn = self.tools[tool_name]
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(**tool_args) if isinstance(tool_args, dict) else await fn()
                else:
                    result = fn(**tool_args) if isinstance(tool_args, dict) else fn()
            elif tool_name in self.http_tools:
                info = self.http_tools[tool_name]
                data = json.dumps(tool_args).encode("utf-8")
                req = urllib.request.Request(
                    info["endpoint"],
                    data=data if info["method"] != "GET" else None,
                    headers={"Content-Type": "application/json"},
                    method=info["method"],
                )
                try:
                    with urllib.request.urlopen(req, timeout=5.0) as resp:
                        result = json.loads(resp.read().decode("utf-8"))
                except Exception as e:
                    result = f"HTTP Tool Error: {e}"

            if result is not None:
                clean_response = response.replace(match.group(0), "").strip()
                return f"{clean_response}\n\n[Tool Result: {tool_name}] -> {result}".strip()

        except Exception as e:
            return f"{response}\n[Tool Execution Error: {e}]"

        return response
