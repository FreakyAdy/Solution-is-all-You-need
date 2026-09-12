"""
PHANTOM CORE — Model Profiles Package
=======================================
Architecture-specific layer maps and hardware detection utilities.
"""

from phantom.model_profiles.auto_detect import (
    ModelArchitecture,
    LayerSpec,
    detect_architecture,
)
from phantom.model_profiles.hardware_detect import (
    HardwareTierProfile,
    GPUInfo,
    SystemInfo,
    detect_hardware,
    print_hardware_report,
)

__all__ = [
    "ModelArchitecture",
    "LayerSpec",
    "detect_architecture",
    "HardwareTierProfile",
    "GPUInfo",
    "SystemInfo",
    "detect_hardware",
    "print_hardware_report",
]