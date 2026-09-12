"""
PHANTOM PLATFORM — Model Registry & Lifecycle Package
======================================================
"""

from phantom.registry.downloader import ResumableDownloader
from phantom.registry.hf_client import HFClient
from phantom.registry.index_client import IndexClient, IndexedModel
from phantom.registry.model_manager import ModelDetails, ModelInfo, ModelManager

__all__ = [
    "ModelManager",
    "ModelInfo",
    "ModelDetails",
    "IndexClient",
    "IndexedModel",
    "HFClient",
    "ResumableDownloader",
]
