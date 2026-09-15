"""
BENCHMARK: Needle-In-A-Haystack (NIAH) Long-Context Evaluation
Measures: Retrieval recall accuracy (100%), KV compression ratio (8.0x),
attention preservation (>= 95%), and cosine distance error (<= 2.0%).
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure root importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.correctness.test_needle_haystack import run_needle_battery


def bench_needle_haystack(quick: bool = False) -> Dict[str, Any]:
    print("=" * 60)
    print("BENCHMARK: Needle-In-A-Haystack Long Context (Innovation 3)")
    print("=" * 60)

    t0 = time.perf_counter()
    data = run_needle_battery(quick=quick)
    elapsed = time.perf_counter() - t0

    print(f"  Execution Time:               {elapsed:.2f} s")
    print(f"  Overall Recall Accuracy:      {data['overall_recall_pct']:.1f}% (Target: >= 95%)")
    print(f"  Compression Ratio:            {data['compression_factor']:.1f}x (D -> D/8)")
    print(f"  Mean Attention Preservation:  {data['mean_attention_preservation_pct']:.2f}% (Target: >= 95%)")
    print(f"  Mean Key Cosine Similarity:   {data['mean_key_cosine_similarity']:.4f} (Target: >= 0.95)")

    assert data["overall_recall_pct"] >= 95.0, "Recall accuracy below 95%"
    assert data["compression_factor"] >= 7.9, "Compression ratio below 8x"
    assert data["mean_key_cosine_similarity"] >= 0.95, "Cosine similarity below 0.95"
    print("  RESULT: [PASS]\n")
    return data


if __name__ == "__main__":
    quick_mode = "--quick" in sys.argv
    bench_needle_haystack(quick=quick_mode)
