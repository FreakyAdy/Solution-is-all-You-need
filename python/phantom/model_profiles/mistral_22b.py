"""
PHANTOM CORE — Mistral 22B (Codestral / Mixtral 8x22B) Model Profile
======================================================================
Fixed architecture constants for Mistral AI's 22B class models.
"""

from __future__ import annotations
from dataclasses import dataclass

# Mistral-22B (e.g., Codestral 22B) architecture constants
NUM_LAYERS = 56
HIDDEN_DIM = 6144
NUM_HEADS = 48
NUM_KV_HEADS = 8   # GQA
HEAD_DIM = 128
FFN_DIM = 16384
VOCAB_SIZE = 32768
MAX_SEQ_LEN = 32768
ROPE_BASE = 1000000.0
ARCH_TYPE = "mistral"

LAYER_SIZE_BYTES_FP16 = int(1.1 * 1024 ** 3)  # ~1.1 GB per layer (FP16)

DEFAULT_SPECTRAL_K_FRACTION = 0.15
DEFAULT_SPARSITY_FRACTION = 0.62
DEFAULT_KV_COMPRESSION_RATIO = 8.0

RECOMMENDED_HOT_LAYERS = 8
RECOMMENDED_WARM_LAYERS = 25


@dataclass
class Mistral22B_Profile:
    """Complete static profile for Mistral 22B class models."""
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
            "name": "mistralai/Codestral-22B-v0.1",
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
            "total_params": 22_000_000_000,
            "layer_size_bytes": LAYER_SIZE_BYTES_FP16,
        }

    def default_spectral_k_map(self) -> dict:
        k = int(self.ffn_dim * DEFAULT_SPECTRAL_K_FRACTION)
        return {i: k for i in range(self.num_layers * 3)}

    def default_gate_config(self) -> dict:
        return {i: [True, 0.5] for i in range(self.num_layers)}
