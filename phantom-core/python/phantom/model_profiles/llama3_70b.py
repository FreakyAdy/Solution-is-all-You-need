"""
PHANTOM CORE — LLaMA-3 70B Model Profile
==========================================
Fixed architecture constants for Meta's LLaMA-3 70B model.
Used as a reference profile when the auto-detector is not available.
"""

from __future__ import annotations
from dataclasses import dataclass

# LLaMA-3 70B architecture constants
NUM_LAYERS = 80
HIDDEN_DIM = 8192
NUM_HEADS = 64
NUM_KV_HEADS = 8   # GQA: 8 KV groups
HEAD_DIM = 128     # HIDDEN_DIM // NUM_HEADS
FFN_DIM = 28672    # Intermediate MLP size
VOCAB_SIZE = 128256
MAX_SEQ_LEN = 8192
ROPE_BASE = 500000.0
ARCH_TYPE = "llama"

# Size per layer in FP16 (bytes)
# Attention: (64+8+8)*128*8192 + 8192*64*128 = ~780 MB
# MLP: 3 * 28672 * 8192 * 2 = ~1.4 GB
# Norm: 2 * 8192 * 2 = ~32 KB
# Total per layer: ~2.2 GB (rough estimate)
LAYER_SIZE_BYTES_FP16 = int(2.2 * 1024 ** 3)  # ~2.2 GB

# Calibration defaults (override with actual calibrated values)
DEFAULT_SPECTRAL_K_FRACTION = 0.15   # Retain 15% of DCT coefficients
DEFAULT_SPARSITY_FRACTION = 0.65     # ~65% of MLP neurons inactive per token
DEFAULT_KV_COMPRESSION_RATIO = 8.0  # KV cache compressed 8x

# Recommended VRAM allocation
RECOMMENDED_HOT_LAYERS = 5   # Keep 5 layers in VRAM (6 GB tier)
RECOMMENDED_WARM_LAYERS = 30  # Keep 30 layers in RAM

@dataclass
class LLaMA3_70B_Profile:
    """Complete static profile for LLaMA-3 70B."""
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
        """Return a dict compatible with Rust PhantomEngine ModelConfig."""
        return {
            "name": "meta-llama/Meta-Llama-3-70B",
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
            "total_params": 70_000_000_000,
            "layer_size_bytes": LAYER_SIZE_BYTES_FP16,
        }

    def default_spectral_k_map(self) -> dict:
        """Default K values for each MLP weight layer."""
        k = int(self.ffn_dim * DEFAULT_SPECTRAL_K_FRACTION)
        return {i: k for i in range(self.num_layers * 3)}  # gate, up, down per layer

    def default_gate_config(self) -> dict:
        """Default sparsity gate config."""
        return {i: [True, 0.5] for i in range(self.num_layers)}
