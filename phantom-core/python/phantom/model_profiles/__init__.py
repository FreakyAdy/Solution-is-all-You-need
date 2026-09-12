"""Model profiles subpackage.

Provides automatic detetcion of model architecture and hardware tier,
plus hand-curated example profiles for common model families.
"""

from __future__ import annotations

from phantom.model_profiles.auto_detect import detect_model
from phantom.model_profiles.hardware_detect import (
    HardwareProfile,
    detect_hardware,
    hardware_tier_name,
)

__all__ = [
    "HardwareProfile",
    "detect_hardware",
    "detect_model",
    "hardware_tier_name",
]