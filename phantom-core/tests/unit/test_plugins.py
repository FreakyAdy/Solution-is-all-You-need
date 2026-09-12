"""Unit tests for Phase 9 Plugin System."""
import asyncio
import pytest
from phantom.plugins import (
    PluginPipeline,
    RAGPlugin,
    ToolRouterPlugin,
    ContextCachePlugin,
    phantom_tool,
    GenerationContext,
)


def test_plugins_end_to_end():
    async def _run():
        pipeline = PluginPipeline()

        # 1. RAG
        rag = RAGPlugin()
        rag.add_document("Quantum Theory", "Quantum mechanics explains subatomic particles.")
        pipeline.register(rag)

        # 2. Tool Router
        router = ToolRouterPlugin()

        @phantom_tool
        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            return f"Sunny 22C in {city}"

        router.register_tool("get_weather", get_weather)
        pipeline.register(router)

        # 3. Context Cache
        cache = ContextCachePlugin()
        pipeline.register(cache)

        ctx = GenerationContext(
            model_id="llama3:70b",
            system_prompt="Long system prompt for testing prefix caching " * 10,
        )

        # Pre-generate
        prompt = "What does Quantum Theory explain?"
        aug_prompt = await pipeline.run_pre_generate(prompt, ctx)
        assert "Quantum mechanics explains subatomic particles" in aug_prompt
        assert "get_weather" in aug_prompt

        # Post-generate tool invocation
        mock_response = (
            'I will check the weather. ```tool_call {"name": "get_weather", "arguments": {"city": "Tokyo"}} ```'
        )
        post_response = await pipeline.run_post_generate(mock_response, ctx)
        assert "[Tool Result: get_weather] -> Sunny 22C in Tokyo" in post_response

        # Prefix Cache hit on second call
        await pipeline.run_pre_generate(prompt, ctx)
        assert ctx.metadata.get("cached_prefix_hit") is True
        assert cache.hits == 1

        print("SUCCESS: Plugin system end-to-end verified!")

    asyncio.run(_run())


if __name__ == "__main__":
    test_plugins_end_to_end()
