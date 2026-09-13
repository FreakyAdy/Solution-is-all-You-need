"""
PHANTOM PLATFORM — Resumable Downloader
=======================================
High-performance chunked file downloader with resumable ranges,
SHA-256 verification, and terminal progress indicators.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ResumableDownloader:
    """Downloads large model files with resume support and progress callbacks."""

    def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB chunks
        self.chunk_size = chunk_size

    def download(
        self,
        url: str,
        dest_path: str | Path,
        expected_sha256: Optional[str] = None,
        progress_cb: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> Path:
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part_path = dest.with_suffix(dest.suffix + ".part")

        downloaded_bytes = 0
        if part_path.exists():
            downloaded_bytes = part_path.stat().st_size

        headers = {
            "User-Agent": "phantom-runtime/1.0",
        }
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token and ("huggingface.co" in url or "hf.co" in url):
            headers["Authorization"] = f"Bearer {hf_token}"

        req = urllib.request.Request(url, headers=headers)
        if downloaded_bytes > 0:
            req.add_header("Range", f"bytes={downloaded_bytes}-")

        try:
            with urllib.request.urlopen(req) as resp:
                total_bytes = downloaded_bytes
                content_len = resp.headers.get("Content-Length")
                if content_len:
                    total_bytes += int(content_len)

                mode = "ab" if downloaded_bytes > 0 else "wb"
                hasher = hashlib.sha256()

                # If resumed, compute hash of existing part
                if downloaded_bytes > 0:
                    with open(part_path, "rb") as f_existing:
                        while chunk := f_existing.read(self.chunk_size):
                            hasher.update(chunk)

                t0 = time.time()
                last_time = t0
                bytes_since_last = 0
                speed = 0.0

                with open(part_path, mode) as f:
                    while True:
                        chunk = resp.read(self.chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        hasher.update(chunk)
                        downloaded_bytes += len(chunk)
                        bytes_since_last += len(chunk)

                        now = time.time()
                        if now - last_time >= 0.5:
                            speed = bytes_since_last / (now - last_time)
                            bytes_since_last = 0
                            last_time = now

                        eta = (total_bytes - downloaded_bytes) / speed if speed > 0 else 0.0
                        if progress_cb:
                            progress_cb(downloaded_bytes, total_bytes, speed, eta)

        except urllib.error.HTTPError as e:
            logger.error("download_failed", error=str(e), url=url, status_code=e.code)
            if e.code == 401:
                raise RuntimeError(
                    f"HTTP Error 401: Unauthorized for {url}\n"
                    "Hugging Face returned 401 Unauthorized. This typically occurs when:\n"
                    "  1. The model repository is private or gated (e.g. Meta-Llama, Gemma) and requires access approval.\n"
                    "  2. To access gated models, set your Hugging Face Access Token:\n"
                    "     Windows (PowerShell):  $env:HF_TOKEN = \"hf_your_token_here\"\n"
                    "     Linux / macOS:         export HF_TOKEN=\"hf_your_token_here\"\n"
                    "     Generate token:        https://huggingface.co/settings/tokens\n"
                    "  3. If not gated, verify that the repository and file name exist on Hugging Face."
                ) from e
            elif e.code == 404:
                raise RuntimeError(
                    f"HTTP Error 404: Model file not found at {url}\n"
                    "Please verify that the model repository and filename exist on Hugging Face."
                ) from e
            raise
        except Exception as e:
            logger.error("download_failed", error=str(e), url=url)
            raise

        # Verify hash
        if expected_sha256:
            calc_hash = hasher.hexdigest()
            if calc_hash.lower() != expected_sha256.lower():
                part_path.unlink(missing_ok=True)
                raise ValueError(
                    f"SHA-256 mismatch! Expected {expected_sha256}, got {calc_hash}"
                )

        if dest.exists():
            dest.unlink()
        part_path.rename(dest)
        logger.info("download_completed", path=str(dest), size_mb=round(downloaded_bytes / 1024**2, 1))
        return dest
