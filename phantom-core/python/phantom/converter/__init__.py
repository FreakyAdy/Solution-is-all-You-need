"""
PHANTOM PLATFORM — Converter Package
=====================================
Format conversion pipeline for transforming GGUF and safetensors
into high-performance .phantomw spectral weights and .phantom packages.
"""

from phantom.converter.format_spec import (
    PHTW_MAGIC,
    PHTW_VERSION,
    PhantomLayerData,
    PhantomLayerReader,
    PhantomLayerWriter,
    PhantomTensorRecord,
)

__all__ = [
    "PHTW_MAGIC",
    "PHTW_VERSION",
    "PhantomLayerData",
    "PhantomLayerReader",
    "PhantomLayerWriter",
    "PhantomTensorRecord",
]
