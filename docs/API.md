# PHANTOM REST & WebSocket API Reference

The PHANTOM API Gateway serves on `http://localhost:11411` by default.

## 1. OpenAI-Compatible Endpoints

### `POST /v1/chat/completions`
Standard OpenAI chat completions with streaming and non-streaming responses.

### `POST /v1/completions`
Text completion endpoint.

### `GET /v1/models`
Returns available models.

### `GET /v1/metrics`
PHANTOM engine telemetry including VRAM/RAM/NVMe residency, tok/sec, and Wraith accuracy.

## 2. Ollama-Compatible Endpoints

### `POST /api/generate`
Drop-in replacement for Ollama's generate API.

### `POST /api/chat`
Drop-in replacement for Ollama's chat API.

### `GET /api/tags`
Lists models in Ollama format.

### `POST /api/pull`
Pulls and converts a model.

### `DELETE /api/delete`
Removes a model.

## 3. PHANTOM Extensions

### `GET /phantom/hardware`
Detected hardware profile and tier ceilings.

### `GET /phantom/models/{id}/layers`
Current layer residency map across memory tiers.

### `POST /phantom/models/{id}/pin-layer`
Force-pin a layer to VRAM or RAM.

### `WS /phantom/metrics/stream`
High-frequency (200ms) WebSocket telemetry stream for real-time dashboards.
