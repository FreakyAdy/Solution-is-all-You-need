"""
PHANTOM PLATFORM — Model Format & Architecture Detection
==========================================================
Unified format identification and deep architecture detection for
GGUF, Safetensors, and HuggingFace directories.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ModelFormat(Enum):
    GGUF = auto()
    SAFETENSORS = auto()
    HUGGINGFACE = auto()
    PHANTOM = auto()
    UNKNOWN = auto()


@dataclass
class ArchitectureSpec:
    """Detailed architectural profile detected from model metadata."""
    family: str                     # "llama", "mistral", "mixtral", "qwen", "phi", "deepseek", "gemma", "falcon"
    version: str                    # e.g., "3.1", "2.5", "v3"
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    ffn_dim: int
    vocab_size: int
    context_length: int
    rope_theta: float = 10000.0
    is_moe: bool = False
    num_experts: int = 0
    num_active_experts: int = 0
    is_mla: bool = False            # DeepSeek Multi-Head Latent Attention
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    norm_type: str = "rmsnorm"      # "rmsnorm", "layernorm"
    activation: str = "silu"        # "silu", "gelu", "swiglu"
    metadata: Dict[str, Any] = field(default_factory=dict)


def detect_format(model_path: str | Path) -> ModelFormat:
    """
    Auto-detect the model format from a file path or directory.
    """
    p = Path(model_path)

    if p.is_file():
        suffix = p.suffix.lower()
        if suffix == ".gguf":
            return ModelFormat.GGUF
        if suffix == ".safetensors":
            return ModelFormat.SAFETENSORS
        if suffix in (".phantom", ".phantomw"):
            return ModelFormat.PHANTOM
        if suffix in (".bin", ".pt", ".pth"):
            return ModelFormat.HUGGINGFACE

    if p.is_dir():
        if (p / "manifest.json").exists() and (p / "weights").exists():
            return ModelFormat.PHANTOM
        if (p / "config.json").exists():
            return ModelFormat.HUGGINGFACE
        if list(p.glob("*.gguf")):
            return ModelFormat.GGUF
        if list(p.glob("*.safetensors")):
            return ModelFormat.SAFETENSORS

    return ModelFormat.UNKNOWN


def detect_architecture_from_hf(config: Dict[str, Any]) -> ArchitectureSpec:
    """Detect architecture specs from HuggingFace config.json."""
    model_type = config.get("model_type", "").lower()
    architectures = [a.lower() for a in config.get("architectures", [])]

    # Defaults
    num_layers = config.get("num_hidden_layers", config.get("n_layer", 32))
    hidden_dim = config.get("hidden_size", config.get("n_embd", 4096))
    num_heads = config.get("num_attention_heads", config.get("n_head", 32))
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    ffn_dim = config.get("intermediate_size", hidden_dim * 4)
    vocab_size = config.get("vocab_size", 32000)
    context_length = config.get("max_position_embeddings", config.get("max_sequence_length", 4096))
    rope_theta = float(config.get("rope_theta", 10000.0))

    family = "llama"
    version = "1"
    is_moe = False
    num_experts = 0
    num_active_experts = 0
    is_mla = False

    if "deepseek" in model_type or any("deepseek" in a for a in architectures):
        family = "deepseek"
        # Check for DeepSeek V2/V3 MLA attention
        if "kv_lora_rank" in config or "q_lora_rank" in config:
            is_mla = True
            version = "v3" if config.get("n_routed_experts", 0) > 64 else "v2"
        if "n_routed_experts" in config:
            is_moe = True
            num_experts = config.get("n_routed_experts", 0)
            num_active_experts = config.get("num_experts_per_tok", 0)
    elif "mixtral" in model_type or any("mixtral" in a for a in architectures):
        family = "mistral"
        version = "mixtral"
        is_moe = True
        num_experts = config.get("num_local_experts", 8)
        num_active_experts = config.get("num_experts_per_tok", 2)
    elif "mistral" in model_type:
        family = "mistral"
        version = "0.3" if ffn_dim == 14336 else "0.1"
    elif "qwen2_moe" in model_type:
        family = "qwen"
        version = "2-moe"
        is_moe = True
        num_experts = config.get("num_experts", 60)
        num_active_experts = config.get("num_experts_per_tok", 4)
    elif "qwen" in model_type:
        family = "qwen"
        version = "2.5" if "2.5" in str(config.get("_name_or_path", "")) else "2"
    elif "phi3" in model_type or "phi-3" in model_type:
        family = "phi"
        version = "3"
    elif "phi" in model_type:
        family = "phi"
        version = "2"
    elif "gemma2" in model_type:
        family = "gemma"
        version = "2"
    elif "gemma" in model_type:
        family = "gemma"
        version = "1"
    elif "falcon" in model_type:
        family = "falcon"
        version = "1"
    elif "llama" in model_type:
        family = "llama"
        if vocab_size >= 128000:
            version = "3.1" if context_length >= 131072 else "3"
        elif vocab_size == 32000:
            version = "2"

    return ArchitectureSpec(
        family=family,
        version=version,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        ffn_dim=ffn_dim,
        vocab_size=vocab_size,
        context_length=context_length,
        rope_theta=rope_theta,
        is_moe=is_moe,
        num_experts=num_experts,
        num_active_experts=num_active_experts,
        is_mla=is_mla,
        q_lora_rank=config.get("q_lora_rank", 0),
        kv_lora_rank=config.get("kv_lora_rank", 0),
        qk_rope_head_dim=config.get("qk_rope_head_dim", 0),
        v_head_dim=config.get("v_head_dim", 0),
        metadata=config,
    )
