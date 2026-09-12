"""
PHANTOM PLATFORM — Plugin System Package
=========================================
"""

from phantom.plugins.base import (
    BasePlugin,
    GenerationContext,
    PhantomPlugin,
    PluginPipeline,
    Rejection,
)
from phantom.plugins.context_cache.plugin import ContextCachePlugin
from phantom.plugins.rag_connector.plugin import RAGPlugin
from phantom.plugins.tool_router.plugin import ToolRouterPlugin, phantom_tool

__all__ = [
    "PhantomPlugin",
    "BasePlugin",
    "GenerationContext",
    "Rejection",
    "PluginPipeline",
    "RAGPlugin",
    "ToolRouterPlugin",
    "ContextCachePlugin",
    "phantom_tool",
]
