"""
PHANTOM PLATFORM — Community Index Client
==========================================
Fetches and searches curated model profiles from local repo or phantom-models.io.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class IndexedModel:
    id: str
    name: str
    family: str
    parameters: str
    context_length: int
    size_gb: float
    sources: Dict[str, Dict[str, Any]]
    community_profiles: List[Dict[str, Any]]
    phantomfile: str


class IndexClient:
    """Client for the PHANTOM curated model registry."""

    INDEX_URL = "https://phantom-models.io/index.json"

    def __init__(self, local_index_path: Optional[Path] = None):
        self.local_path = local_index_path or (
            Path(__file__).parents[3] / "phantom-models" / "index.json"
        )
        self._cache: Optional[Dict[str, Any]] = None

    def load_index(self, force_refresh: bool = False) -> Dict[str, Any]:
        if self._cache is not None and not force_refresh:
            return self._cache

        # Try local first
        if self.local_path.exists():
            try:
                with open(self.local_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                    return self._cache
            except Exception as e:
                logger.warning("local_index_read_failed", error=str(e))

        # Fetch remote
        try:
            req = urllib.request.Request(self.INDEX_URL, headers={"User-Agent": "phantom-cli/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                self._cache = json.loads(resp.read().decode("utf-8"))
                return self._cache
        except Exception as e:
            logger.warning("remote_index_fetch_failed", error=str(e))

        return {"models": []}

    def get_model(self, model_id: str) -> Optional[IndexedModel]:
        index = self.load_index()
        for m in index.get("models", []):
            if m["id"].lower() == model_id.lower():
                return IndexedModel(**m)
        return None

    def search(self, query: str) -> List[IndexedModel]:
        index = self.load_index()
        results = []
        q = query.lower()
        for m in index.get("models", []):
            if q in m["id"].lower() or q in m["name"].lower() or q in m["family"].lower():
                results.append(IndexedModel(**m))
        return results
