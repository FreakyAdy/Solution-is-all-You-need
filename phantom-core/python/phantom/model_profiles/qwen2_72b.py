"""
PHANTOM CORE — Qwen2 72B Model Profile
========================================
Fixed architecture constants for Alibaba's Qwen2 72B model.
"""

from __future__ import annotations
from dataclasses import dataclass

# Qwen2-72B architecture constants
NUM_LAYERS = 80
HIDDEN_DIM = 8192
NUM_HEADS = 64
NUM_KV_HEADS = 8   # GQA
HEAD_DIM = 128
FFN_DIM = 29568    # Slightly larger than LLaMA-3
VOCAB_SIZE = 152064
MAX_SEQ_LEN = 32768
ROPE_BASE = 1000000.0
ARCH_TYPE = "qwen2"

LAYER_SIZE_BYTES_FP16 = int(2.3 * 1024 ** 3)  # ~2.3 GB per layer

DEFAULT_SPECTRAL_K_FRACTION = 0.15
DEFAULT_SPARSITY_FRACTION = 0.63
DEFAULT_KV_COMPRESSION_RATIO = 8.0

RECOMMENDED_HOT_LAYERS = 5
RECOMMENDED_WARM_LAYERS = 30


@dataclass
class Qwen2_72B_Profile:
    """Complete static profile for Qwen2 72B."""
    num_layers: int = NUM_LAYERS
    hidden_dim: int = HIDDEN_DIM
    num_heads: int = NUM_HEADS
    num_kv_heads: int = NUM_KV_HEADS
    head_dim: int = HEAD_DIM
    ffn_dim: int = FFN_DIM
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = MAX_SEQ_LEN
    rope_base: float = ROPE_BASE
    arch_type: str = ARCH_TYPE
    is_gqa: bool = True

    def to_model_config_dict(self) -> dict:
        return {
            "name": "Qwen/Qwen2-72B-Instruct",
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "ffn_dim": self.ffn_dim,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "rope_base": self.rope_base,
            "arch_type": self.arch_type,
            "total_params": 72_000_000_000,
            "layer_size_bytes": LAYER_SIZE_BYTES_FP16,
        }

    def default_spectral_k_map(self) -> dict:
        k = int(self.ffn_dim * DEFAULT_SPECTRAL_K_FRACTION)
        return {i: k for i in range(self.num_layers * 3)}

    def default_gate_config(self) -> dict:
        return {i: [True, 0.5] for i in range(self.num_layers)}
