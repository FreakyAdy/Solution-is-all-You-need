"""
PHANTOM CORE — WebSocket Token Streaming
=========================================
Standalone WebSocket streaming module for real-time token delivery.
Provides both asyncio-based streaming and a simple HTTP SSE fallback.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator, Callable, Optional

import structlog
from fastapi import WebSocket, WebSocketDisconnect

logger = structlog.get_logger(__name__)


class TokenStreamer:
    """
    Manages WebSocket token streaming with backpressure handling.

    Tokens are placed in an asyncio queue by the generation loop
    and drained by the WebSocket sender.
    """

    def __init__(self, max_queue_size: int = 256):
        self._queue: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=max_queue_size)
        self._cancelled = False
        self.tokens_sent = 0
        self.start_time = time.time()

    async def put_token(self, token: str) -> None:
        """Put a token into the stream queue."""
        if not self._cancelled:
            await self._queue.put(token)
            self.tokens_sent += 1

    async def put_done(self) -> None:
        """Signal end of stream."""
        await self._queue.put(None)

    def cancel(self) -> None:
        """Cancel the stream."""
        self._cancelled = True

    async def stream_to_websocket(self, websocket: WebSocket) -> None:
        """
        Drain the token queue and send each token to the WebSocket client.

        Args:
            websocket: Connected FastAPI WebSocket.
        """
        try:
            while True:
                token = await asyncio.wait_for(self._queue.get(), timeout=60.0)
                if token is None:  # End of stream
                    await websocket.send_json({
                        "token": "",
                        "done": True,
                        "tok_per_sec": self._tok_per_sec(),
                    })
                    break

                await websocket.send_json({
                    "token": token,
                    "done": False,
                })

        except asyncio.TimeoutError:
            logger.warning("stream_timeout")
            await websocket.send_json({"error": "Stream timeout", "done": True})
        except WebSocketDisconnect:
            self.cancel()
            logger.info("websocket_client_disconnected")
        except Exception as e:
            logger.error("websocket_stream_error", error=str(e))
            try:
                await websocket.send_json({"error": str(e), "done": True})
            except Exception:
                pass

    async def stream_as_sse(self) -> AsyncGenerator[bytes, None]:
        """
        Drain the token queue and yield Server-Sent Events (SSE) bytes.

        Used as fallback for clients that don't support WebSocket.

        Yields:
            SSE-formatted bytes.
        """
        try:
            while True:
                token = await asyncio.wait_for(self._queue.get(), timeout=60.0)
                if token is None:
                    data = json.dumps({"token": "", "done": True})
                    yield f"data: {data}\n\n".encode()
                    break

                data = json.dumps({"token": token, "done": False})
                yield f"data: {data}\n\n".encode()

        except asyncio.TimeoutError:
            yield b"data: {\"error\": \"timeout\", \"done\": true}\n\n"
        except Exception as e:
            error_data = json.dumps({"error": str(e), "done": True})
            yield f"data: {error_data}\n\n".encode()

    def _tok_per_sec(self) -> float:
        elapsed = max(time.time() - self.start_time, 0.001)
        return round(self.tokens_sent / elapsed, 2)


async def handle_websocket_session(
    websocket: WebSocket,
    generate_fn: Callable[[str, dict], AsyncGenerator[str, None]],
) -> None:
    """
    Handle a complete WebSocket generation session.

    Accepts multiple requests per connection (persistent sessions).

    Args:
        websocket:   Connected WebSocket.
        generate_fn: Async generator function that yields tokens.
                     Signature: (prompt: str, params: dict) -> AsyncGenerator[str, None]
    """
    await websocket.accept()
    logger.info("ws_session_start")

    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")

            if not prompt:
                await websocket.send_json({"error": "Empty prompt", "done": True})
                continue

            params = {
                "max_tokens": data.get("max_tokens", 512),
                "temperature": data.get("temperature", 0.7),
                "top_p": data.get("top_p", 0.9),
                "top_k": data.get("top_k", 40),
            }

            streamer = TokenStreamer()

            # Run generation and streaming concurrently
            async def run_generation():
                try:
                    async for token in generate_fn(prompt, params):
                        await streamer.put_token(token)
                    await streamer.put_done()
                except Exception as e:
                    logger.error("generation_error", error=str(e))
                    await streamer.put_token(f"\n[Error: {e}]")
                    await streamer.put_done()

            await asyncio.gather(
                run_generation(),
                streamer.stream_to_websocket(websocket),
            )

    except WebSocketDisconnect:
        logger.info("ws_session_end_disconnect")
    except Exception as e:
        logger.error("ws_session_error", error=str(e))
