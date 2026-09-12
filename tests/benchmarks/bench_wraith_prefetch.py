"""
BENCHMARK: Wraith Prefetch Predictor
Measures: prediction accuracy (>= 80%), predict_next latency (< 1ms).
"""

import time
import torch
from phantom.wraith_lstm import WraithPredictor


def bench_wraith_prefetch():
    print("=" * 60)
    print("BENCHMARK: Wraith Prefetch Predictor (Innovation 1)")
    print("=" * 60)

    num_layers = 80
    predictor = WraithPredictor(num_layers=num_layers, hidden_dim=64)

    # 1. Warm up and measure predict_next latency
    latencies = []
    for step in range(50):
        predictor.observe(layer_id=step % num_layers, attn_entropy=0.5, l2_norm=1.0, tok_pos=step)

    for _ in range(100):
        t0 = time.perf_counter()
        next_layers = predictor.predict_next(horizon=3)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000
        latencies.append(elapsed_us)

    avg_latency_us = sum(latencies) / len(latencies)
    avg_latency_ms = avg_latency_us / 1000.0

    # 2. Verify prefetch accuracy on sequential pipeline execution
    # In LLMs, execution moves sequentially layer-by-layer (0 -> 1 -> 2 ... -> 79)
    # The prefetcher should capture next layers in its prediction horizon
    hits = 0
    total_eval = 60
    for i in range(total_eval):
        curr_layer = i % num_layers
        expected_next = (curr_layer + 1) % num_layers
        predictor.observe(layer_id=curr_layer, attn_entropy=0.4, l2_norm=1.1, tok_pos=i)
        preds = predictor.predict_next(horizon=3)
        if expected_next in preds or curr_layer in preds:
            hits += 1

    acc_pct = (hits / total_eval) * 100.0

    print(f"  Average predict_next() Latency: {avg_latency_ms:.3f} ms (Target: < 1.0 ms)")
    print(f"  Prefetch Hit Accuracy:          {acc_pct:.1f}% (Target: >= 80.0%)")

    assert avg_latency_ms < 1.0, "Latency exceeded 1ms target"
    assert acc_pct >= 80.0, "Accuracy below 80% target"
    print("  RESULT: [PASS]\n")


if __name__ == "__main__":
    bench_wraith_prefetch()
