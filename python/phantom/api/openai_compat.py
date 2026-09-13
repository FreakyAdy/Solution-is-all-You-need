"""
PHANTOM CORE — OpenAI-Compatible REST API Server
==================================================
Full OpenAI-compatible API server for PHANTOM CORE inference engine.

Implements:
    POST /v1/chat/completions    — Chat completions (streaming + non-streaming)
    POST /v1/completions         — Text completions
    GET  /v1/models              — List available models
    GET  /v1/health              — Health check
    GET  /v1/metrics             — PHANTOM CORE performance metrics (extension)
    WebSocket /v1/stream         — Real-time token streaming

All model communication goes through IPC to the Rust core.
Server handles concurrent requests via asyncio + request queue.
Max concurrent generations: configurable (default 1 for laptop, 3 for desktop).
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# IPC Configuration
# ─────────────────────────────────────────────────────────────────────────────

if os.name == "nt":
    IPC_PATH = r"\\.\pipe\phantom_core_ipc"
else:
    IPC_PATH = "/tmp/phantom_core.sock"

# ─────────────────────────────────────────────────────────────────────────────
# Request/Response models (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="phantom-core-model")
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1)
    max_tokens: int = Field(default=2048, ge=1)
    stream: bool = Field(default=False)
    repetition_penalty: float = Field(default=1.1, ge=1.0)
    stop: Optional[Union[str, List[str]]] = None


class CompletionRequest(BaseModel):
    model: str = Field(default="phantom-core-model")
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)
    stop: Optional[Union[str, List[str]]] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class PhantomMetrics(BaseModel):
    """PHANTOM CORE extension — live performance metrics."""
    vram_mb: float
    vram_total_mb: float
    ram_mb: float
    nvme_mb: float
    layer_residency: Dict[str, List[int]]
    wraith_accuracy_pct: float
    kv_compression_ratio: float
    active_sparsity_pct: float
    tok_per_sec: float
    thermal_state: str
    throttle_active: bool
    uptime_sec: float
    total_requests: int
    active_requests: int


# ─────────────────────────────────────────────────────────────────────────────
# IPC Client
# ─────────────────────────────────────────────────────────────────────────────

class PhantomIPCClient:
    """
    Async IPC client for communicating with the Rust core engine.

    Uses Unix sockets (Linux/macOS) or Named Pipes (Windows).
    Messages are framed with 4-byte little-endian length prefix + MessagePack body.
    """

    def __init__(self, ipc_path: str = IPC_PATH):
        self.ipc_path = ipc_path
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._lock = asyncio.Lock()
        self._connected = False

    async def connect(self) -> None:
        """Connect to the Rust IPC server."""
        try:
            if os.name == "nt":
                # Windows named pipe
                self._reader, self._writer = await asyncio.open_connection(
                    host="127.0.0.1", port=8081  # Fallback TCP for Windows
                )
            else:
                self._reader, self._writer = await asyncio.open_unix_connection(self.ipc_path)
            self._connected = True
            logger.info("ipc_connected", path=self.ipc_path)
        except Exception as e:
            self._connected = False
            logger.debug("ipc_connect_failed_using_engine_fallback", error=str(e))

    async def disconnect(self) -> None:
        """Close the IPC connection."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._connected = False

    async def send_request(self, request: dict) -> dict:
        """
        Send a request dict and receive a response dict via IPC.

        Uses 4-byte LE length prefix framing with JSON encoding.

        Args:
            request: Request dict (e.g. {"type": "generate", "prompt": "..."}).

        Returns:
            Response dict from Rust core.
        """
        if not self._connected:
            await self.connect()

        if not self._connected:
            # Return mock response when Rust core is not running
            return self._mock_response(request)

        async with self._lock:
            try:
                body = json.dumps(request).encode()
                length = struct.pack("<I", len(body))
                self._writer.write(length + body)
                await self._writer.drain()

                # Read response
                len_bytes = await asyncio.wait_for(
                    self._reader.readexactly(4), timeout=30.0
                )
                resp_len = struct.unpack("<I", len_bytes)[0]
                resp_bytes = await asyncio.wait_for(
                    self._reader.readexactly(resp_len), timeout=120.0
                )
                return json.loads(resp_bytes)

            except Exception as e:
                logger.error("ipc_send_failed", error=str(e))
                self._connected = False
                return self._mock_response(request)

    async def stream_generate(
        self,
        request: dict,
    ) -> AsyncGenerator[str, None]:
        """
        Stream token generation via IPC.

        Yields token strings as they are generated by the Rust core.
        """
        if not self._connected:
            await self.connect()

        if not self._connected:
            # Mock streaming when Rust core is not running
            async for token in self._mock_stream(request):
                yield token
            return

        request["stream"] = True
        body = json.dumps(request).encode()
        length = struct.pack("<I", len(body))

        try:
            self._writer.write(length + body)
            await self._writer.drain()

            # Stream tokens until EOS
            while True:
                len_bytes = await asyncio.wait_for(
                    self._reader.readexactly(4), timeout=60.0
                )
                chunk_len = struct.unpack("<I", len_bytes)[0]
                chunk_bytes = await asyncio.wait_for(
                    self._reader.readexactly(chunk_len), timeout=60.0
                )
                chunk = json.loads(chunk_bytes)

                token = chunk.get("token", "")
                done = chunk.get("done", False)

                if token:
                    yield token

                if done:
                    break

        except asyncio.TimeoutError:
            logger.warning("ipc_stream_timeout")
            yield "\n[PHANTOM CORE: Stream timeout]"
        except Exception as e:
            logger.error("ipc_stream_error", error=str(e))
            yield f"\n[PHANTOM CORE Error: {e}]"

    def _mock_response(self, request: dict) -> dict:
        """Generate a mock response when Rust core is unavailable."""
        prompt = request.get("prompt", request.get("messages", [{}])[-1].get("content", ""))
        return {
            "text": f"[PHANTOM CORE — Rust core not running. Prompt received: '{str(prompt)[:100]}']",
            "tokens_generated": 20,
            "tok_per_sec": 0.0,
        }

    async def _mock_stream(self, request: dict) -> AsyncGenerator[str, None]:
        """Mock streaming for when Rust core is unavailable."""
        tokens = [
            "[PHANTOM", " CORE]", " Rust", " core", " not", " running.",
            " Start", " with:", " `phantom-core", " serve`"
        ]
        for token in tokens:
            await asyncio.sleep(0.05)
            yield token

    async def get_metrics(self) -> dict:
        """Get live metrics from the Rust core."""
        try:
            return await self.send_request({"type": "metrics"})
        except Exception:
            return _default_metrics()

    async def get_models(self) -> list:
        """Get list of loaded models."""
        try:
            resp = await self.send_request({"type": "list_models"})
            return resp.get("models", [])
        except Exception:
            return []


def _default_metrics() -> dict:
    """Return default metrics when Rust core is unavailable."""
    return {
        "vram_mb": 0.0,
        "vram_total_mb": 0.0,
        "ram_mb": 0.0,
        "nvme_mb": 0.0,
        "layer_residency": {"vram": [], "ram": [], "nvme": []},
        "wraith_accuracy_pct": 0.0,
        "kv_compression_ratio": 0.0,
        "active_sparsity_pct": 0.0,
        "tok_per_sec": 0.0,
        "thermal_state": "unknown",
        "throttle_active": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Application state
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.ipc_client = PhantomIPCClient()
        self.start_time = time.time()
        self.total_requests = 0
        self.active_requests = 0
        self.loaded_models: List[str] = []
        self.request_semaphore = asyncio.Semaphore(
            int(os.environ.get("PHANTOM_MAX_CONCURRENT", "1"))
        )


app_state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# App lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("phantom_api_starting")
    await app_state.ipc_client.connect()

    # Try to get loaded models
    try:
        app_state.loaded_models = await app_state.ipc_client.get_models()
    except Exception:
        app_state.loaded_models = ["phantom-core-model"]

    yield  # App running

    logger.info("phantom_api_shutdown")
    await app_state.ipc_client.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PHANTOM CORE API",
    description="OpenAI-compatible inference API for the PHANTOM CORE engine.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: format prompt from messages
# ─────────────────────────────────────────────────────────────────────────────

def _format_chat_prompt(messages: List[Message]) -> str:
    """Format chat messages into a single prompt string."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"<|system|>\n{msg.content}</s>")
        elif msg.role == "user":
            parts.append(f"<|user|>\n{msg.content}</s>")
        elif msg.role == "assistant":
            parts.append(f"<|assistant|>\n{msg.content}</s>")
    parts.append("<|assistant|>")
    return "\n".join(parts)


def _estimate_tokens(text: str) -> int:
    """Quick token count estimate (4 chars ≈ 1 token)."""
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/health")
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_state.start_time
    return {
        "status": "ok",
        "engine": "PHANTOM CORE",
        "uptime_sec": round(uptime, 1),
        "ipc_connected": app_state.ipc_client._connected,
        "active_requests": app_state.active_requests,
    }


@app.get("/v1/models")
async def list_models():
    """List available models. OpenAI-compatible response."""
    models = app_state.loaded_models or ["phantom-core-model"]
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": int(app_state.start_time),
                "owned_by": "phantom-core",
                "permission": [],
                "root": m,
                "parent": None,
            }
            for m in models
        ],
    }


@app.get("/v1/metrics", response_model=PhantomMetrics)
async def get_metrics():
    """
    PHANTOM CORE extension endpoint — returns live engine performance metrics.

    Returns:
        PhantomMetrics with VRAM/RAM/NVMe usage, Wraith accuracy,
        KV compression ratio, sparsity %, tok/sec, and thermal state.
    """
    raw = await app_state.ipc_client.get_metrics()
    uptime = time.time() - app_state.start_time

    return PhantomMetrics(
        vram_mb=raw.get("vram_mb", 0.0),
        vram_total_mb=raw.get("vram_total_mb", 0.0),
        ram_mb=raw.get("ram_mb", 0.0),
        nvme_mb=raw.get("nvme_mb", 0.0),
        layer_residency=raw.get("layer_residency", {"vram": [], "ram": [], "nvme": []}),
        wraith_accuracy_pct=raw.get("wraith_accuracy_pct", 0.0),
        kv_compression_ratio=raw.get("kv_compression_ratio", 8.0),
        active_sparsity_pct=raw.get("active_sparsity_pct", 0.0),
        tok_per_sec=raw.get("tok_per_sec", 0.0),
        thermal_state=raw.get("thermal_state", "nominal"),
        throttle_active=raw.get("throttle_active", False),
        uptime_sec=round(uptime, 1),
        total_requests=app_state.total_requests,
        active_requests=app_state.active_requests,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Supports both streaming (stream=true) and non-streaming responses.
    Forwards to Rust core via IPC with full sampling parameters.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    prompt = _format_chat_prompt(request.messages)
    prompt_tokens = _estimate_tokens(prompt)

    ipc_request = {
        "type": "generate",
        "request_id": request_id,
        "prompt": prompt,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "repetition_penalty": request.repetition_penalty,
        "stop_sequences": ([request.stop] if isinstance(request.stop, str) else request.stop) or [],
    }

    app_state.total_requests += 1

    if request.stream:
        async def generate_stream() -> AsyncGenerator[bytes, None]:
            app_state.active_requests += 1
            completion_tokens = 0
            try:
                async with app_state.request_semaphore:
                    async for token in app_state.ipc_client.stream_generate(ipc_request):
                        completion_tokens += 1
                        chunk = ChatCompletionChunk(
                            id=request_id,
                            created=created,
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(content=token),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n".encode()

                # Final chunk with finish_reason
                final_chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n".encode()
                yield b"data: [DONE]\n\n"

            except Exception as e:
                logger.error("stream_error", request_id=request_id, error=str(e))
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n".encode()
            finally:
                app_state.active_requests -= 1

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    else:
        # Non-streaming
        app_state.active_requests += 1
        try:
            async with app_state.request_semaphore:
                response = await app_state.ipc_client.send_request(ipc_request)
        finally:
            app_state.active_requests -= 1

        generated_text = response.get("text", "")
        completion_tokens = response.get("tokens_generated", _estimate_tokens(generated_text))

        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=generated_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    OpenAI-compatible text completions endpoint.

    Accepts a raw prompt string or list of prompts.
    """
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    prompt_text = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    prompt_tokens = _estimate_tokens(prompt_text)

    ipc_request = {
        "type": "generate",
        "request_id": request_id,
        "prompt": prompt_text,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stop_sequences": ([request.stop] if isinstance(request.stop, str) else request.stop) or [],
    }

    app_state.total_requests += 1
    app_state.active_requests += 1
    try:
        async with app_state.request_semaphore:
            response = await app_state.ipc_client.send_request(ipc_request)
    finally:
        app_state.active_requests -= 1

    generated_text = response.get("text", "")
    completion_tokens = response.get("tokens_generated", _estimate_tokens(generated_text))

    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "text": generated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.websocket("/v1/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time token streaming.

    Client sends: {"prompt": "...", "max_tokens": 512, "temperature": 0.7}
    Server streams: {"token": "...", "done": false}
    Server closes: {"token": "", "done": true}
    """
    await websocket.accept()
    logger.info("websocket_connected")

    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")
            if not prompt:
                await websocket.send_json({"error": "No prompt provided"})
                continue

            ipc_request = {
                "type": "generate",
                "prompt": prompt,
                "max_tokens": data.get("max_tokens", 512),
                "temperature": data.get("temperature", 0.7),
                "top_p": data.get("top_p", 0.9),
                "stream": True,
            }

            app_state.total_requests += 1
            app_state.active_requests += 1
            try:
                async with app_state.request_semaphore:
                    async for token in app_state.ipc_client.stream_generate(ipc_request):
                        await websocket.send_json({"token": token, "done": False})

                await websocket.send_json({"token": "", "done": True})

            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({"error": str(e), "done": True})
            finally:
                app_state.active_requests -= 1

    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
    except Exception as e:
        logger.error("websocket_error", error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Exception handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": type(exc).__name__,
                "code": 500,
            }
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(
    host: str = "0.0.0.0",
    port: int = 8080,
    workers: int = 1,
    log_level: str = "info",
) -> None:
    """
    Start the PHANTOM CORE API server.

    Args:
        host:      Bind address (default 0.0.0.0).
        port:      Listen port (default 8080).
        workers:   Number of worker processes.
        log_level: Uvicorn log level.
    """
    import uvicorn

    print("\n╔══════════════════════════════════════════════════╗")
    print("║   PHANTOM CORE API Server — Run the Unreachable  ║")
    print(f"║   Listening on http://{host}:{port}              ║")
    print("╚══════════════════════════════════════════════════╝\n")

    uvicorn.run(
        "phantom.api.openai_compat:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PHANTOM CORE API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    main(host=args.host, port=args.port, log_level=args.log_level)
