"""
Unit tests for PHANTOM Long-Context Needle-In-A-Haystack (NIAH) Evaluation.
Tests autoencoder calibration, memory accounting, embedding generation,
and retrieval fidelity across context windows up to 32K.
"""

import math
import pytest
import torch
from phantom.neural_cache_ae import KVAutoencoder
from tests.correctness.test_needle_haystack import (
    build_calibrated_autoencoder,
    generate_synthetic_haystack_embeddings,
    evaluate_needle_retrieval,
    run_needle_battery,
)


def test_calibrated_autoencoder_architecture():
    """Verify calibrated autoencoder dimensions and 8x compression ratio."""
    head_dim = 128
    ae = build_calibrated_autoencoder(head_dim=head_dim)
    assert ae.head_dim == 128

    x = torch.zeros(16, head_dim)
    x[:, :16] = torch.randn(16, 16).abs() + 0.5
    compressed = ae.encode(x)
    assert compressed.shape == (16, 16)  # 128 // 8 = 16
    recon = ae.decode(compressed)
    assert recon.shape == (16, 128)

    # 8x compression factor
    ratio = x.numel() / compressed.numel()
    assert ratio == 8.0


def test_synthetic_haystack_generation():
    """Verify synthetic haystack shapes, needle placement, and affinity."""
    seq_len = 2048
    head_dim = 128
    depth_pct = 50.0

    query, keys, values, needle_idx = generate_synthetic_haystack_embeddings(
        seq_len=seq_len,
        head_dim=head_dim,
        needle_depth_pct=depth_pct,
        seed=123,
    )

    assert query.shape == (1, head_dim)
    assert keys.shape == (seq_len, head_dim)
    assert values.shape == (seq_len, head_dim)
    assert 0 <= needle_idx < seq_len
    # Verify depth is roughly 50%
    expected_idx = int(seq_len * 0.5)
    assert abs(needle_idx - expected_idx) <= 2


def test_kv_memory_accounting():
    """Verify exact byte and megabyte formulas for uncompressed vs 8x Neural Cache."""
    seq_len = 32768
    head_dim = 128
    num_heads = 8
    num_layers = 32

    res = evaluate_needle_retrieval(
        context_length=seq_len,
        depth_pct=50.0,
        head_dim=head_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    # Uncompressed: 2 (K+V) * 32 layers * 8 heads * 128 dim * 2 bytes = 131,072 bytes/token
    expected_uncompressed_bytes = seq_len * (2 * num_layers * num_heads * head_dim * 2)
    expected_uncompressed_mb = expected_uncompressed_bytes / (1024 * 1024)
    assert math.isclose(res.uncompressed_kv_mb, 4096.0, rel_tol=1e-3)
    assert math.isclose(res.neural_cache_kv_mb, 512.0, rel_tol=1e-3)
    assert res.memory_reduction_factor == 8.0


def test_needle_retrieval_fidelity():
    """Verify needle retrieval accuracy and cosine similarity under 8x compression."""
    res = evaluate_needle_retrieval(
        context_length=4096,
        depth_pct=25.0,
        head_dim=128,
        num_heads=8,
        num_layers=32,
    )

    assert res.baseline_top1_match is True
    assert res.neural_cache_top1_match is True
    assert res.key_cosine_similarity >= 0.95
    assert res.attention_preservation_pct >= 90.0


def test_quick_battery_execution():
    """Verify run_needle_battery executes cleanly in quick mode."""
    results = run_needle_battery(quick=True)
    assert results["total_tests"] == 6  # 2 contexts (4K, 8K) x 3 depths
    assert results["overall_recall_pct"] == 100.0
    assert results["compression_factor"] == 8.0
    assert results["mean_key_cosine_similarity"] >= 0.95
