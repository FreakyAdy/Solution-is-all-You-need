"""PHANTOM CORE — Python orchestration layer.

Handles model loading, hardware detection, calibration, and the Wraith
predictor. Everything runs through the Rust core for inference; this package
produces the calibration profiles that unlock PHANTOM CORE's optimizations.

Subpackages:
    phantom.model_profiles — auto-detection of model architecture & hardware
    phantom.api           — OpenAI-compatible REST + WebSocket server
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]