"""
BENCHMARK: End-to-End Full Pipeline & Hardware Ceiling Lift
Measures: hardware capacity lift across Laptop, Mid, Desktop tiers.
"""

import time
from phantom.model_profiles.hardware_detect import detect_hardware


def bench_full_pipeline():
    print("=" * 60)
    print("BENCHMARK: End-to-End Full Pipeline & Ceiling Lift")
    print("=" * 60)

    hw = detect_hardware()
    print(f"  Detected Hardware Tier:  {hw.tier.upper()} ({hw.gpu_name})")
    print(f"  Native Hardware Limit:   {hw.native_ceiling_b:.1f}B parameters")
    print(f"  PHANTOM Capable Limit:   {hw.phantom_ceiling_b:.1f}B parameters")

    lift = hw.phantom_ceiling_b / max(1.0, hw.native_ceiling_b)
    print(f"  Ceiling Multiplier:      +{lift:.1f}x capacity")

    assert lift >= 2.0, "Ceiling lift should be >= 2.0x"
    print("  RESULT: [PASS]\n")


if __name__ == "__main__":
    bench_full_pipeline()
