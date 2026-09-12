"""
PHANTOM PLATFORM — Tool Router Plugin
======================================
Function calling and MCP tool server routing middleware.
Allows Python functions decorated with @phantom_tool to be invoked
by models during generation.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Callable, Dict, List, Optional

from phantom.plugins.base import BasePlugin, GenerationContext

# Global registry of phantom tools
TOOL_REGISTRY: Dict[str, Callable] = {}


def phantom_tool(fn: Callable) -> Callable:
    """Decorator to register a function as a PHANTOM tool."""
    TOOL_REGISTRY[fn.__name__] = fn
    return fn


class ToolRouterPlugin(BasePlugin):
    name = "tool-router"
    version = "1.0.0"

    def __init__(self, tools: Optional[Dict[str, Callable]] = None):
        self.tools = tools or TOOL_REGISTRY

    def register_tool(self, name: str, fn: Callable):
        self.tools[name] = fn

    def get_tool_definitions(self) -> str:
        """Format registered tools into system prompt instructions."""
        if not self.tools:
            return ""

        schemas = []
        for name, fn in self.tools.items():
            sig = str(inspect.signature(fn))
            doc = fn.__doc__ or "No description provided."
            schemas.append(f"- `{name}{sig}`: {doc.strip()}")

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

            if tool_name in self.tools:
                fn = self.tools[tool_name]
                result = fn(**tool_args) if isinstance(tool_args, dict) else fn()
                # Clean raw JSON call and attach tool execution result
                clean_response = response.replace(match.group(0), "").strip()
                return f"{clean_response}\n\n[Tool Result: {tool_name}] -> {result}".strip()
        except Exception as e:
            return f"{response}\n[Tool Execution Error: {e}]"

        return response
