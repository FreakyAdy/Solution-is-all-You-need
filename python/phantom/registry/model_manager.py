"""
PHANTOM PLATFORM — Model Lifecycle Manager
==========================================
Manages local model repository at ~/.phantom/models/.
Handles pull, list, show, rm, and search across local storage,
the curated PHANTOM index, and HuggingFace Hub.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from phantom.converter.phantom_convert import PhantomConverter
from phantom.registry.downloader import ResumableDownloader
from phantom.registry.hf_client import HFClient
from phantom.registry.index_client import IndexClient, IndexedModel


@dataclass
class ModelInfo:
    id: str
    name: str
    size_str: str
    quantization: str
    context: str
    tok_per_sec: float
    modified_ago: str
    path: str


@dataclass
class ModelDetails:
    id: str
    manifest: Dict[str, Any]
    config: str
    has_calibration: bool
    calibration_stats: Dict[str, Any]
    path: str


class ModelManager:
    """Central model lifecycle engine for PHANTOM."""

    def __init__(self, phantom_home: Optional[Union[str, Path]] = None):
        base = phantom_home or os.environ.get("PHANTOM_HOME", Path.home() / ".phantom")
        self.home = Path(base)
        self.models_dir = self.home / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir = self.home / "downloads"
        self.downloads_dir.mkdir(parents=True, exist_ok=True)

        self.index_client = IndexClient()
        self.hf_client = HFClient()
        self.downloader = ResumableDownloader()

    def list(self, format: Literal["table", "json"] = "table") -> Union[List[ModelInfo], str]:
        """List all locally installed PHANTOM models."""
        models: List[ModelInfo] = []

        for p in self.models_dir.iterdir():
            if not p.is_dir():
                continue
            manifest_file = p / "manifest.json"
            if not manifest_file.exists():
                continue

            try:
                with open(manifest_file, "r") as f:
                    manifest = json.load(f)

                # Total size (include the referenced GGUF for passthrough installs)
                total_bytes = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                gguf_ref = manifest.get("gguf_path")
                if gguf_ref and Path(gguf_ref).is_file():
                    total_bytes += Path(gguf_ref).stat().st_size
                size_gb = total_bytes / (1024**3)
                size_str = f"{size_gb:.1f} GB" if size_gb >= 1.0 else f"{total_bytes / 1024**2:.0f} MB"

                # Modified time
                mtime = manifest_file.stat().st_mtime
                delta_sec = time.time() - mtime
                if delta_sec < 3600:
                    modified = f"{int(delta_sec / 60)} min ago"
                elif delta_sec < 86400:
                    modified = f"{int(delta_sec / 3600)} hours ago"
                else:
                    modified = f"{int(delta_sec / 86400)} days ago"

                # Calib stats
                tok_sec = 0.0
                calib_file = p / "profile" / "calibration.phantom"
                if calib_file.exists():
                    try:
                        with open(calib_file) as cf:
                            cdata = json.load(cf)
                            tok_sec = cdata.get("tok_per_sec", 4.2)
                    except Exception:
                        pass

                models.append(
                    ModelInfo(
                        id=p.name,
                        name=manifest.get("model_family", p.name),
                        size_str=size_str,
                        quantization="SPECTRAL" if manifest.get("spectral_quant") else "BF16",
                        context=f"{int(manifest.get('context_length', 4096) / 1024)}K",
                        tok_per_sec=tok_sec,
                        modified_ago=modified,
                        path=str(p),
                    )
                )
            except Exception as e:
                logger.warning("read_manifest_failed", path=str(p), error=str(e))

        if format == "json":
            return [m.__dict__ for m in models]

        # Table formatting
        if not models:
            return "No models installed. Run 'phantom pull <model>' to download one."

        lines = [
            f"{'NAME':<20} {'SIZE':<10} {'QUANT':<12} {'CONTEXT':<10} {'TOK/SEC':<10} {'MODIFIED':<15}",
            "-" * 80,
        ]
        for m in models:
            lines.append(
                f"{m.id:<20} {m.size_str:<10} {m.quantization:<12} {m.context:<10} {m.tok_per_sec:<10.1f} {m.modified_ago:<15}"
            )
        return "\n".join(lines)

    def show(self, model_id: str) -> ModelDetails:
        """Get complete technical details and profile for a local model."""
        target = self.models_dir / model_id
        if not target.exists():
            raise FileNotFoundError(f"Model '{model_id}' is not installed in {self.models_dir}")

        manifest = {}
        if (target / "manifest.json").exists():
            with open(target / "manifest.json") as f:
                manifest = json.load(f)

        config_str = ""
        if (target / "config.toml").exists():
            with open(target / "config.toml") as f:
                config_str = f.read()

        calib_stats = {}
        has_calib = False
        calib_file = target / "profile" / "calibration.phantom"
        if calib_file.exists():
            has_calib = True
            try:
                with open(calib_file) as f:
                    calib_stats = json.load(f)
            except Exception:
                pass

        return ModelDetails(
            id=model_id,
            manifest=manifest,
            config=config_str,
            has_calibration=has_calib,
            calibration_stats=calib_stats,
            path=str(target),
        )

    def rm(self, model_id: str, force: bool = False) -> None:
        """Remove a model directory from local library."""
        target = self.models_dir / model_id
        if not target.exists():
            raise FileNotFoundError(f"Model '{model_id}' does not exist")
        shutil.rmtree(target)
        logger.info("model_removed", model_id=model_id)

    def pull(
        self,
        model_ref: str,
        quantization: str = "Q4_K_M",
        no_calibrate: bool = False,
        skip_convert: bool = False,
        progress_cb: Optional[Callable[[str, float, str], None]] = None,
    ) -> Path:
        """
        Pull a model by resolving reference, downloading GGUF, and running conversion.
        """
        dest_model_dir = self.models_dir / model_ref.replace(":", "-").replace("/", "_")
        if dest_model_dir.exists() and (dest_model_dir / "manifest.json").exists():
            logger.info("model_already_exists", path=str(dest_model_dir))
            return dest_model_dir

        # 1. Resolve source URL
        source_url = None
        source_sha256 = None
        filename = f"{dest_model_dir.name}.gguf"

        # Check local file
        local_gguf: Optional[Path] = None
        if Path(model_ref).exists() and Path(model_ref).is_file():
            local_gguf = Path(model_ref)
            dest_model_dir = self.models_dir / f"{local_gguf.stem.replace('.', '-')}-{quantization}"
            filename = local_gguf.name
        elif model_ref.startswith("http://") or model_ref.startswith("https://"):
            source_url = model_ref
            local_gguf = self.downloads_dir / filename
        else:
            # Check community index
            indexed = self.index_client.get_model(model_ref)
            if indexed and quantization in indexed.sources:
                info = indexed.sources[quantization]
                source_url = info["url"]
                source_sha256 = info.get("sha256")
                filename = info.get("filename", filename)
                local_gguf = self.downloads_dir / filename
            else:
                # Reuse a previously downloaded GGUF for this model (no network needed).
                last = model_ref.rsplit("/", 1)[-1]
                reuse: Optional[Path] = None
                for cand in sorted(self.downloads_dir.glob("*.gguf")):
                    if last.lower() in cand.name.lower():
                        if quantization.lower() in cand.name.lower():
                            reuse = cand
                            break
                        if reuse is None:
                            reuse = cand
                if reuse is not None:
                    local_gguf = reuse

            if local_gguf is None:
                # Search HuggingFace Hub
                files = self.hf_client.list_gguf_files(model_ref)
                matched = [f for f in files if quantization.lower() in f.filename.lower()]
                selected = matched[0] if matched else (files[0] if files else None)

                if selected:
                    source_url = selected.url
                    local_gguf = self.downloads_dir / selected.filename
                else:
                    raise ValueError(f"Could not resolve model reference: {model_ref}")

        # Reuse an existing passthrough install pointing at this exact GGUF instead
        # of starting a fresh (often multi-hour) conversion of the same file.
        if not skip_convert and local_gguf is not None:
            target = local_gguf.resolve()
            for cand in sorted(self.models_dir.iterdir()):
                manifest_file = cand / "manifest.json"
                if not manifest_file.is_file():
                    continue
                try:
                    with open(manifest_file, "r") as f:
                        man = json.load(f)
                except Exception:
                    continue
                if (
                    man.get("mode") == "passthrough"
                    and Path(man.get("gguf_path", "")).resolve() == target
                ):
                    logger.info("reusing_passthrough_install", path=str(cand), model=model_ref)
                    return cand

        # 2. Download if needed
        if source_url and not local_gguf.exists():
            if progress_cb:
                progress_cb("downloading", 0.0, f"Starting download: {filename}")

            def _dl_cb(cur, total, speed, eta):
                pct = (cur / total * 100.0) if total > 0 else 0.0
                speed_mb = speed / (1024 * 1024)
                if progress_cb:
                    progress_cb("downloading", pct, f"{speed_mb:.1f} MB/s | ETA: {int(eta)}s")

            self.downloader.download(source_url, local_gguf, expected_sha256=source_sha256, progress_cb=_dl_cb)

        # 3. Convert to PHANTOM Native Format
        if skip_convert:
            # Passthrough mode: just create manifest pointing to GGUF
            dest_model_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "version": 1,
                "model_id": model_ref,
                "mode": "passthrough",
                "gguf_path": str(local_gguf),
            }
            with open(dest_model_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            return dest_model_dir

        converter = PhantomConverter(
            input_path=local_gguf,
            output_dir=dest_model_dir,
            spectral_quant=True,
            calibrate=not no_calibrate,
            progress_cb=progress_cb,
        )
        return converter.convert()

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search both community index and Hugging Face Hub."""
        results: List[Dict[str, Any]] = []

        # 1. Local community index
        for m in self.index_client.search(query):
            results.append({
                "source": "phantom_registry",
                "id": m.id,
                "name": m.name,
                "parameters": m.parameters,
                "context": f"{int(m.context_length / 1024)}K",
                "quants": list(m.sources.keys()),
            })

        # 2. Hugging Face
        hf_matches = self.hf_client.search_models(query, limit=5)
        for h in hf_matches:
            results.append({
                "source": "huggingface",
                "id": h.get("id", ""),
                "name": h.get("id", ""),
                "parameters": "Unknown",
                "context": "Auto",
                "quants": ["GGUF"],
            })

        return results
