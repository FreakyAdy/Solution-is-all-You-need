"""
PHANTOM PLATFORM — HuggingFace Hub Integration
===============================================
Discovers and locates GGUF model files from HuggingFace Hub repositories.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class HFModelFile:
    filename: str
    size_bytes: int
    url: str
    quant_type: str


class HFClient:
    """Queries Hugging Face API for GGUF model files."""

    API_BASE = "https://huggingface.co/api"

    def list_gguf_files(self, repo_id: str) -> List[HFModelFile]:
        """List all .gguf files in a repository with direct download URLs."""
        url = f"{self.API_BASE}/models/{repo_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "phantom-cli/1.0"})

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                siblings = data.get("siblings", [])
                
                gguf_files: List[HFModelFile] = []
                for s in siblings:
                    fn = s.get("rfilename", "")
                    if fn.lower().endswith(".gguf"):
                        # Extract quant type from filename
                        quant = "UNKNOWN"
                        for q in ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "Q6_K", "F16", "BF16", "Q4_0", "Q4_1"]:
                            if q.lower() in fn.lower():
                                quant = q
                                break

                        dl_url = f"https://huggingface.co/{repo_id}/resolve/main/{fn}"
                        gguf_files.append(
                            HFModelFile(
                                filename=fn,
                                size_bytes=s.get("size", 0),
                                url=dl_url,
                                quant_type=quant,
                            )
                        )
                return gguf_files
        except Exception as e:
            logger.warning("hf_repo_query_failed", repo=repo_id, error=str(e))
            return []

    def search_models(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search HuggingFace for GGUF models."""
        encoded = urllib.parse.quote(f"{query} gguf")
        url = f"{self.API_BASE}/models?search={encoded}&limit={limit}&full=false"
        req = urllib.request.Request(url, headers={"User-Agent": "phantom-cli/1.0"})

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning("hf_search_failed", query=query, error=str(e))
            return []
