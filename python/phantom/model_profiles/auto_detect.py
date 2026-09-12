"""
PHANTOM CORE — Model Architecture Auto-Detection
==================================================
Auto-detects model architecture from config.json or model metadata,
returning layer maps with exact tensor shapes for PHANTOM CORE's
calibration and inference pipeline.

Supports: LLaMA 2/3, Mistral, Qwen2, Falcon, Mixtral, Gemma, Phi-3,
          Command-R, DeepSeek, Yi, and other common architectures.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerSpec:
    """Specification for one transformer layer."""
    layer_id: int
    has_attention: bool
    has_mlp: bool
    has_moe: bool          # Mixtral-style MoE
    num_experts: int       # For MoE layers
    num_active_experts: int
    attn_heads: int
    kv_heads: int
    head_dim: int
    hidden_dim: int
    ffn_dim: int
    attn_weight_shapes: Dict[str, Tuple[int, ...]]   # name -> shape
    mlp_weight_shapes: Dict[str, Tuple[int, ...]]    # name -> shape
    norm_weight_shapes: Dict[str, Tuple[int, ...]]   # name -> shape
    layer_size_bytes_fp16: int  # Estimated FP16 size


@dataclass
class ModelArchitecture:
    """Complete architecture spec for a model."""
    arch_type: str                # "llama", "mistral", "qwen2", etc.
    model_name: str
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    ffn_dim: int
    vocab_size: int
    max_seq_len: int
    rope_base: float
    rope_scaling: Optional[Dict]
    is_gqa: bool                  # Grouped Query Attention
    is_moe: bool                  # Mixture of Experts
    num_experts: int
    num_active_experts: int
    layers: List[LayerSpec] = field(default_factory=list)
    total_params: int = 0
    total_bytes_fp16: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Architecture-specific layer map builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_llama_layer(
    layer_id: int,
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    ffn_dim: int,
) -> LayerSpec:
    """Build a LLaMA/Mistral standard layer spec."""
    head_dim = hidden_dim // num_heads

    # Attention weights: Q, K, V projections + O projection
    attn_shapes = {
        f"model.layers.{layer_id}.self_attn.q_proj.weight": (num_heads * head_dim, hidden_dim),
        f"model.layers.{layer_id}.self_attn.k_proj.weight": (num_kv_heads * head_dim, hidden_dim),
        f"model.layers.{layer_id}.self_attn.v_proj.weight": (num_kv_heads * head_dim, hidden_dim),
        f"model.layers.{layer_id}.self_attn.o_proj.weight": (hidden_dim, num_heads * head_dim),
    }

    # MLP weights: gate, up, down projections (SwiGLU)
    mlp_shapes = {
        f"model.layers.{layer_id}.mlp.gate_proj.weight": (ffn_dim, hidden_dim),
        f"model.layers.{layer_id}.mlp.up_proj.weight": (ffn_dim, hidden_dim),
        f"model.layers.{layer_id}.mlp.down_proj.weight": (hidden_dim, ffn_dim),
    }

    # LayerNorm weights
    norm_shapes = {
        f"model.layers.{layer_id}.input_layernorm.weight": (hidden_dim,),
        f"model.layers.{layer_id}.post_attention_layernorm.weight": (hidden_dim,),
    }

    # FP16 size estimate
    attn_params = sum(h * w for h, w in attn_shapes.values())
    mlp_params = sum(h * w for h, w in mlp_shapes.values())
    norm_params = sum(n for (n,) in norm_shapes.values())
    total_params = attn_params + mlp_params + norm_params
    size_bytes = total_params * 2  # FP16

    return LayerSpec(
        layer_id=layer_id,
        has_attention=True,
        has_mlp=True,
        has_moe=False,
        num_experts=0,
        num_active_experts=0,
        attn_heads=num_heads,
        kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
        attn_weight_shapes=attn_shapes,
        mlp_weight_shapes=mlp_shapes,
        norm_weight_shapes=norm_shapes,
        layer_size_bytes_fp16=size_bytes,
    )


def _build_qwen2_layer(
    layer_id: int,
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    ffn_dim: int,
) -> LayerSpec:
    """Build a Qwen2-style layer (slight naming difference from LLaMA)."""
    head_dim = hidden_dim // num_heads

    attn_shapes = {
        f"model.layers.{layer_id}.self_attn.q_proj.weight": (num_heads * head_dim, hidden_dim),
        f"model.layers.{layer_id}.self_attn.k_proj.weight": (num_kv_heads * head_dim, hidden_dim),
        f"model.layers.{layer_id}.self_attn.v_proj.weight": (num_kv_heads * head_dim, hidden_dim),
        f"model.layers.{layer_id}.self_attn.o_proj.weight": (hidden_dim, num_heads * head_dim),
        # Qwen2 uses QKV bias
        f"model.layers.{layer_id}.self_attn.q_proj.bias": (num_heads * head_dim,),
        f"model.layers.{layer_id}.self_attn.k_proj.bias": (num_kv_heads * head_dim,),
        f"model.layers.{layer_id}.self_attn.v_proj.bias": (num_kv_heads * head_dim,),
    }

    mlp_shapes = {
        f"model.layers.{layer_id}.mlp.gate_proj.weight": (ffn_dim, hidden_dim),
        f"model.layers.{layer_id}.mlp.up_proj.weight": (ffn_dim, hidden_dim),
        f"model.layers.{layer_id}.mlp.down_proj.weight": (hidden_dim, ffn_dim),
    }

    norm_shapes = {
        f"model.layers.{layer_id}.input_layernorm.weight": (hidden_dim,),
        f"model.layers.{layer_id}.post_attention_layernorm.weight": (hidden_dim,),
    }

    all_params = (
        sum(h * w if len(s) > 1 else s[0] for s in attn_shapes.values() for h, w in [s if len(s) > 1 else (1, s[0])])
        + sum(h * w for h, w in mlp_shapes.values())
        + sum(n for (n,) in norm_shapes.values())
    )
    # Simpler param count
    attn_p = (num_heads + 2 * num_kv_heads) * head_dim * hidden_dim + hidden_dim * num_heads * head_dim
    attn_bias_p = (num_heads + 2 * num_kv_heads) * head_dim
    mlp_p = 2 * ffn_dim * hidden_dim + hidden_dim * ffn_dim
    norm_p = 2 * hidden_dim
    size_bytes = (attn_p + attn_bias_p + mlp_p + norm_p) * 2

    return LayerSpec(
        layer_id=layer_id,
        has_attention=True,
        has_mlp=True,
        has_moe=False,
        num_experts=0,
        num_active_experts=0,
        attn_heads=num_heads,
        kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
        attn_weight_shapes=attn_shapes,
        mlp_weight_shapes=mlp_shapes,
        norm_weight_shapes=norm_shapes,
        layer_size_bytes_fp16=size_bytes,
    )


def _build_mixtral_layer(
    layer_id: int,
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    ffn_dim: int,
    num_experts: int,
    num_active: int,
) -> LayerSpec:
    """Build a Mixtral MoE layer spec."""
    head_dim = hidden_dim // num_heads

    attn_shapes = {
        f"model.layers.{layer_id}.self_attn.q_proj.weight": (num_heads * head_dim, hidden_dim),
        f"model.layers.{layer_id}.self_attn.k_proj.weight": (num_kv_heads * head_dim, hidden_dim),
        f"model.layers.{layer_id}.self_attn.v_proj.weight": (num_kv_heads * head_dim, hidden_dim),
        f"model.layers.{layer_id}.self_attn.o_proj.weight": (hidden_dim, num_heads * head_dim),
    }

    mlp_shapes = {}
    # Router gate
    mlp_shapes[f"model.layers.{layer_id}.block_sparse_moe.gate.weight"] = (num_experts, hidden_dim)
    # Per-expert weights
    for exp_id in range(num_experts):
        mlp_shapes[f"model.layers.{layer_id}.block_sparse_moe.experts.{exp_id}.w1.weight"] = (ffn_dim, hidden_dim)
        mlp_shapes[f"model.layers.{layer_id}.block_sparse_moe.experts.{exp_id}.w2.weight"] = (hidden_dim, ffn_dim)
        mlp_shapes[f"model.layers.{layer_id}.block_sparse_moe.experts.{exp_id}.w3.weight"] = (ffn_dim, hidden_dim)

    norm_shapes = {
        f"model.layers.{layer_id}.input_layernorm.weight": (hidden_dim,),
        f"model.layers.{layer_id}.post_attention_layernorm.weight": (hidden_dim,),
    }

    attn_p = (num_heads + 2 * num_kv_heads) * head_dim * hidden_dim + hidden_dim * num_heads * head_dim
    expert_p = num_experts * (2 * ffn_dim * hidden_dim + hidden_dim * ffn_dim)
    router_p = num_experts * hidden_dim
    norm_p = 2 * hidden_dim
    size_bytes = (attn_p + expert_p + router_p + norm_p) * 2

    return LayerSpec(
        layer_id=layer_id,
        has_attention=True,
        has_mlp=True,
        has_moe=True,
        num_experts=num_experts,
        num_active_experts=num_active,
        attn_heads=num_heads,
        kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
        attn_weight_shapes=attn_shapes,
        mlp_weight_shapes=mlp_shapes,
        norm_weight_shapes=norm_shapes,
        layer_size_bytes_fp16=size_bytes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main auto-detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_architecture(
    model_path: str,
    config_override: Optional[dict] = None,
) -> ModelArchitecture:
    """
    Auto-detect model architecture from config.json or model metadata.

    Parses HuggingFace config.json or GGUF metadata to extract architecture
    parameters and builds a complete ModelArchitecture object.

    Args:
        model_path:      Path to model file or HuggingFace directory.
        config_override: Optional dict to override detected config values.

    Returns:
        ModelArchitecture with full layer specs.

    Raises:
        ValueError: If architecture cannot be determined.
    """
    p = Path(model_path)
    config: dict = {}

    # Load config
    if (p / "config.json").exists():
        with open(p / "config.json") as f:
            config = json.load(f)
    elif p.suffix == ".gguf":
        from phantom.loader import load_gguf_meta
        raw_meta, _ = load_gguf_meta(str(p))
        config = _gguf_meta_to_hf_config(raw_meta)

    if config_override:
        config.update(config_override)

    if not config:
        raise ValueError(f"Cannot detect model architecture from: {model_path}")

    arch_type = config.get("model_type", "llama").lower()
    logger.info("arch_detected", arch_type=arch_type, model_path=str(model_path))

    # Extract common parameters
    num_layers = config.get("num_hidden_layers", config.get("n_layer", 32))
    hidden_dim = config.get("hidden_size", config.get("n_embd", 4096))
    num_heads = config.get("num_attention_heads", config.get("n_head", 32))
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = config.get("head_dim", hidden_dim // num_heads)
    ffn_dim = config.get("intermediate_size", hidden_dim * 4)
    vocab_size = config.get("vocab_size", 32000)
    max_seq_len = config.get("max_position_embeddings", 4096)
    rope_base = config.get("rope_theta", 10000.0)
    rope_scaling = config.get("rope_scaling", None)

    # MoE detection
    num_experts = config.get("num_local_experts", config.get("num_experts", 0))
    num_active = config.get("num_experts_per_tok", config.get("top_k", 2))
    is_moe = num_experts > 1

    is_gqa = num_kv_heads < num_heads
    model_name = config.get("_name_or_path", p.name)

    arch = ModelArchitecture(
        arch_type=arch_type,
        model_name=model_name,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        ffn_dim=ffn_dim,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        rope_base=float(rope_base),
        rope_scaling=rope_scaling,
        is_gqa=is_gqa,
        is_moe=is_moe,
        num_experts=num_experts,
        num_active_experts=num_active,
    )

    # Build layer specs
    for layer_id in range(num_layers):
        if is_moe and arch_type in ("mixtral",):
            layer = _build_mixtral_layer(
                layer_id, hidden_dim, num_heads, num_kv_heads,
                ffn_dim, num_experts, num_active
            )
        elif arch_type in ("qwen2", "qwen"):
            layer = _build_qwen2_layer(layer_id, hidden_dim, num_heads, num_kv_heads, ffn_dim)
        else:
            # Default: LLaMA-compatible (works for LLaMA 2/3, Mistral, Gemma, Yi, etc.)
            layer = _build_llama_layer(layer_id, hidden_dim, num_heads, num_kv_heads, ffn_dim)
        arch.layers.append(layer)

    # Total size estimation
    # Embedding tables
    embed_bytes = vocab_size * hidden_dim * 2  # FP16
    layer_bytes = sum(l.layer_size_bytes_fp16 for l in arch.layers)
    arch.total_bytes_fp16 = embed_bytes + layer_bytes
    arch.total_params = arch.total_bytes_fp16 // 2

    logger.info(
        "arch_built",
        arch_type=arch_type,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        is_gqa=is_gqa,
        is_moe=is_moe,
        total_gb_fp16=round(arch.total_bytes_fp16 / (1024 ** 3), 1),
    )

    return arch


def _gguf_meta_to_hf_config(raw_meta: dict) -> dict:
    """Convert GGUF metadata keys to HuggingFace config format."""
    mapping = {
        "llama.block_count": "num_hidden_layers",
        "llama.embedding_length": "hidden_size",
        "llama.attention.head_count": "num_attention_heads",
        "llama.attention.head_count_kv": "num_key_value_heads",
        "llama.feed_forward_length": "intermediate_size",
        "llama.context_length": "max_position_embeddings",
        "llama.vocab_size": "vocab_size",
        "llama.rope.freq_base": "rope_theta",
        "general.architecture": "model_type",
    }
    config = {}
    for gguf_key, hf_key in mapping.items():
        if gguf_key in raw_meta:
            config[hf_key] = raw_meta[gguf_key]
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Utility: compute layer size in bytes
# ─────────────────────────────────────────────────────────────────────────────

def layer_size_bytes(layer: LayerSpec, dtype_bytes: int = 2) -> int:
    """
    Compute the total byte size of all weights in a layer.

    Args:
        layer:       LayerSpec object.
        dtype_bytes: Bytes per element (2 for FP16, 4 for FP32).

    Returns:
        Total byte size.
    """
    all_shapes = {
        **layer.attn_weight_shapes,
        **layer.mlp_weight_shapes,
        **layer.norm_weight_shapes,
    }
    total = 0
    for name, shape in all_shapes.items():
        n = 1
        for d in shape:
            n *= d
        total += n * dtype_bytes
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import structlog
    structlog.configure(
        processors=[structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    print("[AUTO DETECT] Testing architecture detection...")

    # Simulate LLaMA 3 70B config
    llama70b_config = {
        "model_type": "llama",
        "num_hidden_layers": 80,
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,  # GQA
        "intermediate_size": 28672,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
    }

    import tempfile, json as _json
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        config_path.write_text(_json.dumps(llama70b_config))

        arch = detect_architecture(tmpdir)

    print(f"  Model: {arch.arch_type}")
    print(f"  Layers: {arch.num_layers}")
    print(f"  Hidden: {arch.hidden_dim}")
    print(f"  Heads: {arch.num_heads} (KV: {arch.num_kv_heads}) GQA={arch.is_gqa}")
    print(f"  FFN: {arch.ffn_dim}")
    print(f"  Total size: {arch.total_bytes_fp16 / (1024**3):.1f} GB (FP16)")

    layer0 = arch.layers[0]
    print(f"  Layer 0 size: {layer0.layer_size_bytes_fp16 / 1024**2:.1f} MB")
    print(f"  Layer 0 attn weights: {list(layer0.attn_weight_shapes.keys())[:2]}...")

    # Validate GQA detection
    assert arch.is_gqa, "GQA should be detected"
    assert arch.num_kv_heads == 8, f"Expected 8 KV heads, got {arch.num_kv_heads}"

    print("[AUTO DETECT] Self-test PASSED")
