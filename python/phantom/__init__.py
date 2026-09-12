"""
PHANTOM CORE — Python Package
==============================
Universal Hardware-Transcendent LLM Inference Engine.

"The GPU doesn't know its limits until you show it what it's missing."

Run the Unreachable.
"""

__version__ = "0.1.0"
__author__ = "PHANTOM CORE Project"

from phantom.loader import (
    ModelFormat,
    ModelMeta,
    TensorMeta,
    detect_format,
    load_model_meta,
    stream_layers,
    load_model_for_calibration,
)

from phantom.wraith_lstm import WraithPredictor, warm_up_from_log
from phantom.neural_cache_ae import (
    KVAutoencoder,
    KVCollector,
    train_kv_autoencoder,
    export_for_cuda as export_kv_ae,
)
from phantom.spectral_analyzer import (
    analyze_model_spectral,
    dct_1d_type2,
    idct_1d_type2,
    select_k_for_layer,
)
from phantom.calibrate import CalibrationPipeline

__all__ = [
    # Core classes
    "WraithPredictor",
    "KVAutoencoder",
    "KVCollector",
    "CalibrationPipeline",
    # Model loading
    "ModelFormat",
    "ModelMeta",
    "TensorMeta",
    "detect_format",
    "load_model_meta",
    "stream_layers",
    "load_model_for_calibration",
    # Calibration functions
    "train_kv_autoencoder",
    "export_kv_ae",
    "analyze_model_spectral",
    "warm_up_from_log",
    # Low-level
    "dct_1d_type2",
    "idct_1d_type2",
    "select_k_for_layer",
]