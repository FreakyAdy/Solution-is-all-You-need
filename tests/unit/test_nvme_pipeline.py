"""
Unit Tests for PHANTOM AsyncTilePagingEngine & NVMe Streaming Pipeline
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import pytest

from phantom.instrumentation.nvme_pipeline import AsyncTilePagingEngine, TileDescriptor


def test_async_engine_roundtrip():
    """Verify byte-for-byte fidelity of tiles written and read through AsyncTilePagingEngine."""
    with tempfile.TemporaryDirectory() as tmpdir:
        swap_path = Path(tmpdir) / "test_swap.bin"
        engine = AsyncTilePagingEngine(swap_path, tile_size_bytes=1024 * 1024)

        test_data = b"PHANTOM_PAGE_LAYER_WEIGHT_DATA_SAMPLE" * 1000
        size = len(test_data)
        offset = 4096

        # Write
        written = engine.write_tile(offset, test_data)
        assert written == size

        # Synchronous read
        read_back = engine.read_tile_synchronous(offset, size)
        assert read_back == test_data

        engine.close()


def test_async_double_buffer_prefetch():
    """Verify asynchronous prefetching into ping-pong buffers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        swap_path = Path(tmpdir) / "test_swap_async.bin"
        engine = AsyncTilePagingEngine(swap_path, tile_size_bytes=64 * 1024)

        tile_size = 4096
        data_layer_0 = b"\x11" * tile_size
        data_layer_1 = b"\x22" * tile_size

        engine.write_tile(0, data_layer_0)
        engine.write_tile(4096, data_layer_1)

        # Submit prefetch for layer 0
        engine.submit_prefetch_tile(layer_id=0, offset=0, size=tile_size)
        view_0 = engine.await_tile(layer_id=0)
        assert bytes(view_0) == data_layer_0

        # Submit prefetch for layer 1 (double buffer ping-pong)
        engine.submit_prefetch_tile(layer_id=1, offset=4096, size=tile_size)
        view_1 = engine.await_tile(layer_id=1)
        assert bytes(view_1) == data_layer_1

        engine.close()


def test_fused_swiglu_idct_emulate():
    """Verify numerical execution of fused SwiGLU + iDCT emulation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        swap_path = Path(tmpdir) / "test_swap_swiglu.bin"
        engine = AsyncTilePagingEngine(swap_path, tile_size_bytes=1024 * 1024)

        coeffs = np.ones((64, 32), dtype=np.int8)
        act_x = np.ones(32, dtype=np.float32) * 0.1

        out = engine.fused_swiglu_idct_emulate(coeffs, act_x, k_retained=32)
        assert out.shape == (32,)
        assert np.all(np.isfinite(out))
        assert engine.fused_decompression_savings_bytes > 0

        engine.close()


def test_benchmark_pipeline_runs():
    """Verify benchmark_pipeline executes cleanly and reports valid statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        swap_path = Path(tmpdir) / "test_swap_bench.bin"
        engine = AsyncTilePagingEngine(swap_path, tile_size_bytes=1024 * 1024)

        res = engine.benchmark_pipeline(n_tiles=3, tile_size_mb=1.0)
        assert res["n_tiles"] == 3
        assert res["sync_throughput_mean_gbps"] > 0.0
        assert res["effective_throughput_fused_gbps"] > 0.0
        assert res["fused_compression_factor"] == 2.0

        engine.close()
