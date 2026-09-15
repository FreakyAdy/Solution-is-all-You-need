"""
PHANTOM Instrumentation Subsystem
=================================
High-precision byte accounting, execution tracing, and environment fingerprinting.
"""

from phantom.instrumentation.byte_counter import ByteCounter, get_global_byte_counter
from phantom.instrumentation.fingerprint import get_environment_fingerprint
from phantom.instrumentation.nvme_pipeline import AsyncTilePagingEngine, TileDescriptor

__all__ = [
    "ByteCounter",
    "get_global_byte_counter",
    "get_environment_fingerprint",
    "AsyncTilePagingEngine",
    "TileDescriptor",
]
