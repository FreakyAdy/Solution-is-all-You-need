"""
PHANTOM PLATFORM — Extensible Plugin System Base
=================================================
Lifecycle interceptors for inference middleware:
  - pre_request:   inspect/modify or reject incoming request
  - pre_generate:  modify prompt (e.g. inject RAG context)
  - on_token:      inspect/filter/modify generated tokens
  - post_generate: post-process response
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union


@dataclass
class GenerationContext:
    model_id: str
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: str = ""
    history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rejection:
    status_code: int = 400
    detail: str = "Request rejected by plugin"


class PhantomPlugin(Protocol):
    """Protocol implemented by all PHANTOM middleware plugins."""

    name: str
    version: str

    async def pre_request(self, request: Dict[str, Any]) -> Union[Dict[str, Any], Rejection]:
        ...

    async def pre_generate(self, prompt: str, context: GenerationContext) -> str:
        ...

    async def on_token(self, token: str, context: GenerationContext) -> Optional[str]:
        ...

    async def post_generate(self, response: str, context: GenerationContext) -> str:
        ...


class BasePlugin:
    """Convenience base class providing no-op default implementations."""

    name: str = "base"
    version: str = "1.0.0"

    async def pre_request(self, request: Dict[str, Any]) -> Union[Dict[str, Any], Rejection]:
        return request

    async def pre_generate(self, prompt: str, context: GenerationContext) -> str:
        return prompt

    async def on_token(self, token: str, context: GenerationContext) -> Optional[str]:
        return token

    async def post_generate(self, response: str, context: GenerationContext) -> str:
        return response


class PluginPipeline:
    """Executes registered plugins sequentially through the 4 intercept points."""

    def __init__(self, plugins: Optional[List[BasePlugin]] = None):
        self.plugins: List[BasePlugin] = plugins or []

    def register(self, plugin: BasePlugin):
        self.plugins.append(plugin)

    async def run_pre_request(self, req: Dict[str, Any]) -> Union[Dict[str, Any], Rejection]:
        current = req
        for p in self.plugins:
            res = await p.pre_request(current)
            if isinstance(res, Rejection):
                return res
            current = res
        return current

    async def run_pre_generate(self, prompt: str, context: GenerationContext) -> str:
        current = prompt
        for p in self.plugins:
            current = await p.pre_generate(current, context)
        return current

    async def run_on_token(self, token: str, context: GenerationContext) -> Optional[str]:
        current: Optional[str] = token
        for p in self.plugins:
            if current is None:
                break
            current = await p.on_token(current, context)
        return current

    async def run_post_generate(self, response: str, context: GenerationContext) -> str:
        current = response
        for p in self.plugins:
            current = await p.post_generate(current, context)
        return current
