"""
PHANTOM PLATFORM — Loader Package
==================================
Multi-format model weight loading system with native GGUF SIMD dequantization,
Safetensors streaming, HuggingFace directory parsing, and architecture detection.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch

from phantom.loader.format_detect import (
    ArchitectureSpec,
    ModelFormat,
    detect_architecture_from_hf,
    detect_format,
)
from phantom.loader.gguf_loader import (
    GGUFLoader,
    GGUFMetadata,
    GGUFQuantType,
    GGUFTensorInfo,
)
from phantom.loader.safetensors_loader import SafetensorsLoader

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class TensorMeta:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    layer_id: Optional[int] = None
    layer_type: Optional[str] = None  # "attention", "mlp", "norm", "embed"
    size_bytes: int = 0


@dataclass
class ModelMeta:
    name: str
    format: ModelFormat
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    ffn_dim: int
    vocab_size: int
    max_seq_len: int
    arch_type: str
    total_params: int
    total_bytes: int
    tensors: List[TensorMeta] = field(default_factory=list)


def _annotate_tensors(tensors: List[TensorMeta]) -> None:
    for t in tensors:
        name = t.name.lower()
        layer_match = re.search(r"(?:layers?|blocks?|transformer\.h)\.(\d+)", name)
        if layer_match:
            t.layer_id = int(layer_match.group(1))

        if any(kw in name for kw in ["self_attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj"]):
            t.layer_type = "attention"
        elif any(kw in name for kw in ["mlp", "ffn", "feed_forward", "fc1", "fc2", "gate_proj", "up_proj", "down_proj"]):
            t.layer_type = "mlp"
        elif any(kw in name for kw in ["norm", "ln_"]):
            t.layer_type = "norm"
        elif any(kw in name for kw in ["embed", "wte", "wpe", "lm_head"]):
            t.layer_type = "embed"


def load_model_meta(model_path: Union[str, Path]) -> ModelMeta:
    p = Path(model_path)
    fmt = detect_format(p)
    t0 = time.time()

    if fmt == ModelFormat.GGUF:
        loader = GGUFLoader(p if p.is_file() else list(p.glob("*.gguf"))[0])
        arch = loader.meta.architecture
        tensors = [
            TensorMeta(
                name=t.name,
                shape=t.shape,
                dtype=t.quant_type.name.lower(),
                size_bytes=t.size_bytes,
            )
            for t in loader.meta.tensors.values()
        ]
        _annotate_tensors(tensors)
        total_params = sum(int(np.prod(t.shape)) for t in tensors)
        total_bytes = sum(t.size_bytes for t in tensors)

        return ModelMeta(
            name=loader.meta.metadata_kv.get("general.name", p.stem),
            format=fmt,
            num_layers=arch.num_layers,
            hidden_dim=arch.hidden_dim,
            num_heads=arch.num_heads,
            num_kv_heads=arch.num_kv_heads,
            ffn_dim=arch.ffn_dim,
            vocab_size=arch.vocab_size,
            max_seq_len=arch.context_length,
            arch_type=arch.family,
            total_params=total_params,
            total_bytes=total_bytes,
            tensors=tensors,
        )

    elif fmt in (ModelFormat.SAFETENSORS, ModelFormat.HUGGINGFACE):
        config_path = p / "config.json" if p.is_dir() else p.parent / "config.json"
        config = {}
        if config_path.exists():
            import json
            with open(config_path) as f:
                config = json.load(f)
        arch = detect_architecture_from_hf(config)

        tensors = []
        if fmt == ModelFormat.SAFETENSORS:
            st = SafetensorsLoader(p if p.is_file() else list(p.glob("*.safetensors"))[0])
            for name in st.get_tensor_names():
                info = st.header[name]
                tensors.append(
                    TensorMeta(
                        name=name,
                        shape=tuple(info["shape"]),
                        dtype=info["dtype"].lower(),
                        size_bytes=info["data_offsets"][1] - info["data_offsets"][0],
                    )
                )

        _annotate_tensors(tensors)
        total_params = sum(int(np.prod(t.shape)) for t in tensors)
        total_bytes = sum(t.size_bytes for t in tensors)

        return ModelMeta(
            name=config.get("_name_or_path", p.stem),
            format=fmt,
            num_layers=arch.num_layers,
            hidden_dim=arch.hidden_dim,
            num_heads=arch.num_heads,
            num_kv_heads=arch.num_kv_heads,
            ffn_dim=arch.ffn_dim,
            vocab_size=arch.vocab_size,
            max_seq_len=arch.context_length,
            arch_type=arch.family,
            total_params=total_params,
            total_bytes=total_bytes,
            tensors=tensors,
        )

    raise ValueError(f"Unsupported model format for {model_path}")


def stream_layers(
    model_path: Union[str, Path],
    target_dtype: torch.dtype = torch.bfloat16,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    p = Path(model_path)
    fmt = detect_format(p)

    if fmt == ModelFormat.GGUF:
        loader = GGUFLoader(p if p.is_file() else list(p.glob("*.gguf"))[0])
        total = len(loader.meta.tensors)
        for i, (name, tensor) in enumerate(loader.iter_tensors()):
            if progress_cb:
                progress_cb(i, total, name)
            yield name, tensor.to(target_dtype)
    elif fmt == ModelFormat.SAFETENSORS:
        st = SafetensorsLoader(p if p.is_file() else list(p.glob("*.safetensors"))[0])
        names = st.get_tensor_names()
        total = len(names)
        for i, name in enumerate(names):
            if progress_cb:
                progress_cb(i, total, name)
            yield name, st.load_tensor(name, target_dtype)


def load_model_for_calibration(
    model_path: str,
    device: str = "cpu",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> Tuple[Any, Any]:
    """Load model and tokenizer using HuggingFace Transformers for calibration."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto" if device == "auto" else device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        logger.warning("hf_load_failed", error=str(e))
        return None, None


__all__ = [
    "ModelFormat",
    "ArchitectureSpec",
    "TensorMeta",
    "ModelMeta",
    "GGUFLoader",
    "GGUFMetadata",
    "GGUFQuantType",
    "GGUFTensorInfo",
    "SafetensorsLoader",
    "detect_format",
    "detect_architecture_from_hf",
    "load_model_meta",
    "stream_layers",
    "load_model_for_calibration",
]
