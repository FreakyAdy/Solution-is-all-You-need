"""
Unit tests for PHANTOM Virtual Hardware & Model Profiler.
Verifies mathematical accuracy of layer residency distribution,
MoE sparse active compute calculations, bus transfer latencies,
and bottleneck diagnostics without any local weights downloaded.
"""

import pytest
from phantom.model_profiles.hardware_simulator import (
    HARDWARE_PRESETS,
    KNOWN_MODELS,
    HardwareProfile,
    ModelSpec,
    resolve_model_spec,
    simulate_model_execution,
)


def test_resolve_known_models():
    """Verify known model resolutions."""
    m_32b = resolve_model_spec("qwen2.5-coder-32b")
    assert m_32b.total_params == 32.76
    assert m_32b.active_params == 32.76
    assert m_32b.is_moe is False

    m_moe = resolve_model_spec("qwen3-30b-a3b")
    assert m_moe.total_params == 30.5
    assert m_moe.active_params == 3.3
    assert m_moe.is_moe is True
    assert m_moe.num_experts == 128
    assert m_moe.num_active_experts == 8

    m_70b = resolve_model_spec("llama-3-70b")
    assert m_70b.total_params == 70.6
    assert m_70b.num_layers == 80


def test_resolve_custom_model_heuristics():
    """Verify heuristic parameter parsing for unknown model strings."""
    m_custom_moe = resolve_model_spec("custom-8x7b-v1")
    assert m_custom_moe.is_moe is True
    assert m_custom_moe.num_experts == 8
    assert m_custom_moe.active_params == 14.0

    m_custom_dense = resolve_model_spec("my-model-13b")
    assert m_custom_dense.is_moe is False
    assert m_custom_dense.total_params == 13.0


def test_simulate_dense_32b_on_rtx4050():
    """Verify simulation of 32B Dense on RTX 4050 laptop matches empirical metrics."""
    res = simulate_model_execution("qwen2.5-coder-32b", "rtx4050-laptop")

    # Layer distribution: must split between VRAM and RAM, 0 in NVMe
    assert res.vram_layers > 0
    assert res.ram_layers > 0
    assert res.nvme_layers == 0
    assert res.vram_layers + res.ram_layers == res.model.num_layers

    # Performance
    assert 2.5 <= res.tok_per_sec <= 4.5
    assert res.bottleneck == "DDR5 System RAM Bandwidth Bound (Optimal Dual-Tier Offload)"
    assert res.scale_multiplier_vs_vram > 3.0


def test_simulate_moe_30b_active_compute():
    """Verify MoE 30B achieves ~4x faster throughput and 10x lower FLOPs than 32B Dense."""
    res_dense = simulate_model_execution("qwen2.5-coder-32b", "rtx4050-laptop")
    res_moe = simulate_model_execution("qwen3-30b-a3b", "rtx4050-laptop")

    # FLOPs comparison: MoE has ~10x fewer FLOPs per token
    assert res_moe.compute_gflops_per_token < res_dense.compute_gflops_per_token / 8.0

    # Active memory traffic: MoE requires significantly less memory bus traffic per token
    assert res_moe.active_transfer_gb_per_token < res_dense.active_transfer_gb_per_token / 3.0

    # Decoding speed: MoE runs substantially faster on the same laptop
    assert res_moe.tok_per_sec >= res_dense.tok_per_sec * 2.5
    assert res_moe.tok_per_sec >= 8.0


def test_simulate_70b_nvme_spillover():
    """Verify 70B model on 6GB VRAM + 24GB RAM properly diagnoses NVMe disk swap bottleneck."""
    res = simulate_model_execution("llama-3-70b", "rtx4050-laptop")

    # Must overflow into NVMe swap on 24GB RAM + 6GB VRAM
    assert res.nvme_layers > 0
    assert "NVMe" in res.bottleneck
    assert res.tok_per_sec < 1.5


def test_simulate_colab_t4():
    """Verify Google Colab T4 (15GB VRAM) preset simulation."""
    res_8b = simulate_model_execution("llama-3.1-8b", "colab-t4")
    # 8B fits completely in 15GB VRAM
    assert res_8b.vram_layers == res_8b.model.num_layers
    assert res_8b.ram_layers == 0
    assert res_8b.nvme_layers == 0
    assert "100% Native GPU Residency" in res_8b.bottleneck
    assert res_8b.tok_per_sec >= 30.0


def test_simulate_rtx4090_desktop():
    """Verify high-end desktop RTX 4090 (24GB VRAM, 64GB RAM) handles 70B with zero NVMe spillover."""
    res = simulate_model_execution("llama-3-70b", "rtx4090-desktop")
    assert res.nvme_layers == 0
    assert res.vram_layers > 40
    assert res.tok_per_sec >= 3.0
