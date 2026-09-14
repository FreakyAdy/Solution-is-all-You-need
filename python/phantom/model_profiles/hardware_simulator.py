"""
PHANTOM CORE — Virtual Hardware & Model Profiler
=================================================
Simulates model layer residency, memory hierarchy tiering (VRAM -> RAM -> NVMe),
bus transfer latencies, and token decoding throughput across arbitrary hardware
configurations and model architectures with ZERO disk storage overhead.

Supports:
- Dense models (135M, 3B, 7B, 8B, 14B, 32B, 70B, 72B)
- Mixture of Experts (MoE) architectures (Mixtral 8x7B, Qwen3-30B-A3B, DeepSeek-V3)
- Hardware presets (RTX 4050, 4060, 4070, 4090, Colab T4, Apple Silicon, custom)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class HardwareProfile:
    """Hardware specifications for simulation."""
    id: str
    name: str
    vram_gb: float
    vram_bandwidth_gbps: float
    ram_gb: float
    ram_bandwidth_gbps: float
    os_reserved_ram_gb: float
    nvme_gb: float
    nvme_read_gbps: float
    pcie_bandwidth_gbps: float
    compute_tflops_fp16: float


@dataclass
class ModelSpec:
    """Model architecture specification for zero-disk simulation."""
    id: str
    name: str
    total_params: float       # in Billions (e.g. 32.76)
    active_params: float      # in Billions (e.g. 3.3 for MoE, 32.76 for Dense)
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    is_moe: bool = False
    num_experts: int = 1
    num_active_experts: int = 1
    default_context: int = 4096


@dataclass
class SimulationResult:
    """Empirical and theoretical simulation result."""
    model: ModelSpec
    hardware: HardwareProfile
    quantization: str
    context_length: int

    # Memory allocation (GB)
    total_weight_gb: float
    active_weight_gb_per_token: float
    kv_cache_raw_gb: float
    kv_cache_compressed_gb: float   # with PHANTOM 8x Neural Cache

    # Layer Distribution
    vram_layers: int
    ram_layers: int
    nvme_layers: int

    vram_weight_gb: float
    ram_weight_gb: float
    nvme_weight_gb: float

    # Usable / Remaining Capacity
    vram_used_gb: float
    vram_free_gb: float
    ram_used_gb: float
    ram_free_gb: float

    # Performance Estimates
    compute_gflops_per_token: float
    active_transfer_gb_per_token: float
    token_latency_sec: float
    tok_per_sec: float
    ttft_warm_sec: float
    ttft_cold_sec: float

    # Architectural Assessment
    bottleneck: str
    memory_tier_status: str
    scale_multiplier_vs_vram: float
    is_supported: bool
    warnings: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Standard Hardware Presets
# ─────────────────────────────────────────────────────────────────────────────

HARDWARE_PRESETS: Dict[str, HardwareProfile] = {
    "rtx4050-laptop": HardwareProfile(
        id="rtx4050-laptop",
        name="NVIDIA GeForce RTX 4050 Laptop (6GB GDDR6, 24GB DDR5)",
        vram_gb=6.0,
        vram_bandwidth_gbps=192.0,      # 96-bit GDDR6 @ 16 Gbps
        ram_gb=24.0,
        ram_bandwidth_gbps=48.0,       # Effective dual-channel DDR5 host AVX2/AVX-512 matrix bandwidth
        os_reserved_ram_gb=6.5,
        nvme_gb=500.0,
        nvme_read_gbps=4.5,            # PCIe Gen4 NVMe tile read rate
        pcie_bandwidth_gbps=7.87,      # PCIe Gen4 x4 laptop link
        compute_tflops_fp16=18.0,
    ),
    "rtx4060-laptop": HardwareProfile(
        id="rtx4060-laptop",
        name="NVIDIA GeForce RTX 4060 Laptop (8GB GDDR6, 16GB DDR5)",
        vram_gb=8.0,
        vram_bandwidth_gbps=256.0,      # 128-bit GDDR6
        ram_gb=16.0,
        ram_bandwidth_gbps=38.0,
        os_reserved_ram_gb=5.5,
        nvme_gb=500.0,
        nvme_read_gbps=4.5,
        pcie_bandwidth_gbps=15.75,     # PCIe Gen4 x8
        compute_tflops_fp16=24.0,
    ),
    "rtx4070-desktop": HardwareProfile(
        id="rtx4070-desktop",
        name="NVIDIA GeForce RTX 4070 Desktop (12GB GDDR6X, 32GB DDR5)",
        vram_gb=12.0,
        vram_bandwidth_gbps=504.0,
        ram_gb=32.0,
        ram_bandwidth_gbps=55.0,
        os_reserved_ram_gb=7.0,
        nvme_gb=1000.0,
        nvme_read_gbps=6.0,
        pcie_bandwidth_gbps=31.5,      # PCIe Gen4 x16
        compute_tflops_fp16=40.0,
    ),
    "rtx4090-desktop": HardwareProfile(
        id="rtx4090-desktop",
        name="NVIDIA GeForce RTX 4090 Desktop (24GB GDDR6X, 64GB DDR5)",
        vram_gb=24.0,
        vram_bandwidth_gbps=1008.0,
        ram_gb=64.0,
        ram_bandwidth_gbps=65.0,
        os_reserved_ram_gb=8.0,
        nvme_gb=2000.0,
        nvme_read_gbps=7.0,
        pcie_bandwidth_gbps=31.5,
        compute_tflops_fp16=82.5,
    ),
    "colab-t4": HardwareProfile(
        id="colab-t4",
        name="Google Colab Free Tier (Nvidia T4 15GB, 12.7GB Host RAM)",
        vram_gb=15.0,
        vram_bandwidth_gbps=320.0,      # GDDR6 256-bit
        ram_gb=12.7,
        ram_bandwidth_gbps=25.0,       # Cloud virtualized DDR4
        os_reserved_ram_gb=2.5,
        nvme_gb=100.0,
        nvme_read_gbps=1.8,            # Cloud ephemeral disk
        pcie_bandwidth_gbps=15.75,     # PCIe Gen3 x16
        compute_tflops_fp16=65.0,
    ),
    "apple-m3-pro": HardwareProfile(
        id="apple-m3-pro",
        name="Apple M3 Pro (18GB Unified Memory)",
        vram_gb=14.0,                  # Usable unified slice
        vram_bandwidth_gbps=150.0,
        ram_gb=18.0,
        ram_bandwidth_gbps=150.0,
        os_reserved_ram_gb=4.0,
        nvme_gb=512.0,
        nvme_read_gbps=5.0,
        pcie_bandwidth_gbps=150.0,     # Zero-copy unified fabric
        compute_tflops_fp16=18.0,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Known Model Architectures
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_MODELS: Dict[str, ModelSpec] = {
    "smollm-135m": ModelSpec(
        id="smollm-135m",
        name="SmolLM-135M-Instruct",
        total_params=0.135,
        active_params=0.135,
        num_layers=30,
        hidden_dim=576,
        num_heads=9,
        num_kv_heads=3,
        is_moe=False,
        default_context=2048,
    ),
    "llama-3.2-3b": ModelSpec(
        id="llama-3.2-3b",
        name="Llama-3.2-3B-Instruct",
        total_params=3.21,
        active_params=3.21,
        num_layers=28,
        hidden_dim=3072,
        num_heads=24,
        num_kv_heads=8,
        is_moe=False,
        default_context=4096,
    ),
    "qwen2.5-7b": ModelSpec(
        id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        total_params=7.61,
        active_params=7.61,
        num_layers=28,
        hidden_dim=3584,
        num_heads=28,
        num_kv_heads=4,
        is_moe=False,
        default_context=4096,
    ),
    "llama-3.1-8b": ModelSpec(
        id="llama-3.1-8b",
        name="Llama-3.1-8B-Instruct",
        total_params=8.03,
        active_params=8.03,
        num_layers=32,
        hidden_dim=4096,
        num_heads=32,
        num_kv_heads=8,
        is_moe=False,
        default_context=4096,
    ),
    "qwen2.5-14b": ModelSpec(
        id="qwen2.5-14b",
        name="Qwen2.5-14B-Instruct",
        total_params=14.7,
        active_params=14.7,
        num_layers=48,
        hidden_dim=5120,
        num_heads=40,
        num_kv_heads=8,
        is_moe=False,
        default_context=4096,
    ),
    "qwen3-30b-a3b": ModelSpec(
        id="qwen3-30b-a3b",
        name="Qwen3-30B-A3B (MoE Sparse)",
        total_params=30.5,
        active_params=3.3,           # Only 8 of 128 experts active
        num_layers=48,
        hidden_dim=4096,
        num_heads=32,
        num_kv_heads=8,
        is_moe=True,
        num_experts=128,
        num_active_experts=8,
        default_context=4096,
    ),
    "mixtral-8x7b": ModelSpec(
        id="mixtral-8x7b",
        name="Mixtral-8x7B-Instruct (MoE Sparse)",
        total_params=46.7,
        active_params=12.9,          # 2 of 8 experts active
        num_layers=32,
        hidden_dim=4096,
        num_heads=32,
        num_kv_heads=8,
        is_moe=True,
        num_experts=8,
        num_active_experts=2,
        default_context=4096,
    ),
    "qwen2.5-coder-32b": ModelSpec(
        id="qwen2.5-coder-32b",
        name="Qwen2.5-Coder-32B-Instruct (100% Dense)",
        total_params=32.76,
        active_params=32.76,         # All 32.76B compute on every token
        num_layers=64,
        hidden_dim=5120,
        num_heads=40,
        num_kv_heads=8,
        is_moe=False,
        default_context=4096,
    ),
    "llama-3-70b": ModelSpec(
        id="llama-3-70b",
        name="Llama-3-70B-Instruct (100% Dense)",
        total_params=70.6,
        active_params=70.6,
        num_layers=80,
        hidden_dim=8192,
        num_heads=64,
        num_kv_heads=8,
        is_moe=False,
        default_context=4096,
    ),
    "qwen2.5-72b": ModelSpec(
        id="qwen2.5-72b",
        name="Qwen2.5-72B-Instruct (100% Dense)",
        total_params=72.7,
        active_params=72.7,
        num_layers=80,
        hidden_dim=8192,
        num_heads=64,
        num_kv_heads=8,
        is_moe=False,
        default_context=4096,
    ),
    "deepseek-v3": ModelSpec(
        id="deepseek-v3",
        name="DeepSeek-V3 (671B MoE Sparse)",
        total_params=671.0,
        active_params=37.0,          # 8 of 256 routed experts + shared
        num_layers=61,
        hidden_dim=7168,
        num_heads=128,
        num_kv_heads=128,
        is_moe=True,
        num_experts=256,
        num_active_experts=8,
        default_context=4096,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Quantization Specs
# ─────────────────────────────────────────────────────────────────────────────

QUANT_BITS: Dict[str, float] = {
    "Q4_K_M": 4.5,     # ~4.5 bits per weight with block scales
    "Q4_0": 4.1,
    "Q5_K_M": 5.5,
    "Q8_0": 8.5,
    "FP16": 16.0,
    "BF16": 16.0,
}


def resolve_model_spec(model_ref: str) -> ModelSpec:
    """Resolve a model name or string to a ModelSpec."""
    key = model_ref.lower().strip().replace(":", "-").replace("/", "_")
    for k, spec in KNOWN_MODELS.items():
        if k in key or key in k:
            return spec

    # Fallback heuristic parser for parameter notation (e.g., "7b", "32b", "70b", "8x7b")
    import re
    moe_match = re.search(r"(\d+)x(\d+)b", key)
    if moe_match:
        num_exp = int(moe_match.group(1))
        exp_size = float(moe_match.group(2))
        total = num_exp * exp_size * 0.82
        active = 2 * exp_size
        return ModelSpec(
            id=model_ref,
            name=f"Custom MoE {model_ref}",
            total_params=round(total, 1),
            active_params=round(active, 1),
            num_layers=32,
            hidden_dim=4096,
            num_heads=32,
            num_kv_heads=8,
            is_moe=True,
            num_experts=num_exp,
            num_active_experts=2,
        )

    param_match = re.search(r"(\d+(?:\.\d+)?)b", key)
    if param_match:
        p = float(param_match.group(1))
        layers = 80 if p >= 60 else (64 if p >= 30 else (48 if p >= 12 else (32 if p >= 6 else 24)))
        dim = 8192 if p >= 60 else (5120 if p >= 30 else 4096)
        return ModelSpec(
            id=model_ref,
            name=f"Generic {p}B Model",
            total_params=p,
            active_params=p,
            num_layers=layers,
            hidden_dim=dim,
            num_heads=max(8, dim // 128),
            num_kv_heads=8,
            is_moe=False,
        )

    # Ultimate fallback: 32B model
    return KNOWN_MODELS["qwen2.5-coder-32b"]


def simulate_model_execution(
    model_ref: str,
    hardware_preset: str = "rtx4050-laptop",
    custom_hw: Optional[HardwareProfile] = None,
    quantization: str = "Q4_K_M",
    context_length: int = 4096,
) -> SimulationResult:
    """Simulate model residency, bus latencies, and performance across memory tiers."""
    model = resolve_model_spec(model_ref)
    hw = custom_hw or HARDWARE_PRESETS.get(hardware_preset, HARDWARE_PRESETS["rtx4050-laptop"])

    bits_per_param = QUANT_BITS.get(quantization.upper(), 4.5)
    bytes_per_param = bits_per_param / 8.0

    # Total model weight size in GB
    total_weight_gb = (model.total_params * 1e9 * bytes_per_param) / (1024**3)

    # Active weights per token:
    if model.is_moe:
        # MoE active weights: shared attention layers + active expert weights
        attention_fraction = 0.25
        active_expert_fraction = (model.num_active_experts / max(1, model.num_experts)) * 0.75
        active_weight_ratio = attention_fraction + active_expert_fraction
        active_weight_gb_per_token = total_weight_gb * active_weight_ratio
    else:
        # Dense: 100% of weights are read/computed every token
        active_weight_gb_per_token = total_weight_gb

    # KV Cache calculations (FP16 bytes = 2 bytes/element)
    head_dim = model.hidden_dim // max(1, model.num_heads)
    # 2 (keys + values) * layers * kv_heads * head_dim * context * 2 bytes
    kv_raw_bytes = 2 * model.num_layers * model.num_kv_heads * head_dim * context_length * 2
    kv_cache_raw_gb = kv_raw_bytes / (1024**3)
    # PHANTOM 8x Neural Cache autoencoder compression
    kv_cache_compressed_gb = kv_cache_raw_gb / 8.0

    # Execution buffer overhead (scratchpad, activation memory, CUDA context)
    cuda_context_overhead_gb = 0.65
    vram_scratchpad_gb = 0.35 + (0.15 if model.total_params >= 30 else 0.05)

    # Compute available memory budgets
    usable_vram_gb = max(0.1, hw.vram_gb - cuda_context_overhead_gb - kv_cache_compressed_gb - vram_scratchpad_gb)
    usable_ram_gb = max(0.5, hw.ram_gb - hw.os_reserved_ram_gb)

    # Layer sizing
    gb_per_layer = total_weight_gb / max(1, model.num_layers)

    # Layer Distribution across hierarchy
    vram_layers = min(model.num_layers, max(0, int(usable_vram_gb / gb_per_layer)))
    vram_weight_gb = vram_layers * gb_per_layer

    remaining_layers_after_vram = model.num_layers - vram_layers
    ram_layers = min(remaining_layers_after_vram, max(0, int(usable_ram_gb / gb_per_layer)))
    ram_weight_gb = ram_layers * gb_per_layer

    nvme_layers = remaining_layers_after_vram - ram_layers
    nvme_weight_gb = nvme_layers * gb_per_layer

    # Real memory allocation accounting
    vram_used_gb = vram_weight_gb + cuda_context_overhead_gb + kv_cache_compressed_gb + vram_scratchpad_gb
    vram_free_gb = max(0.0, hw.vram_gb - vram_used_gb)

    ram_used_gb = ram_weight_gb + hw.os_reserved_ram_gb
    ram_free_gb = max(0.0, hw.ram_gb - ram_used_gb)

    # Compute load: 2 FLOPs per active parameter
    compute_gflops_per_token = 2.0 * model.active_params

    # Effective latency calculation per generated token
    # 1. Compute time on GPU tensor cores:
    # effective TFLOPs achieved in generation batch size 1 is typically ~15-20% of theoretical peak
    effective_tflops = hw.compute_tflops_fp16 * 0.18
    t_compute_sec = (compute_gflops_per_token / 1000.0) / max(0.1, effective_tflops)

    # 2. Memory bus transfer time per token:
    # How much of active weights reside in each tier?
    ratio_vram = vram_layers / max(1, model.num_layers)
    ratio_ram = ram_layers / max(1, model.num_layers)
    ratio_nvme = nvme_layers / max(1, model.num_layers)

    active_vram_gb = active_weight_gb_per_token * ratio_vram
    active_ram_gb = active_weight_gb_per_token * ratio_ram
    active_nvme_gb = active_weight_gb_per_token * ratio_nvme

    # Latency components:
    # VRAM bandwidth transfer
    t_vram_sec = active_vram_gb / max(1.0, hw.vram_bandwidth_gbps)
    # RAM bandwidth transfer (CPU evaluates host RAM layers in-place via SIMD; only activations cross PCIe)
    t_ram_sec = active_ram_gb / max(1.0, hw.ram_bandwidth_gbps)
    # NVMe tile paging latency (PHANTOM 64MB compressed tiles via Wraith prefetcher)
    t_nvme_sec = active_nvme_gb / max(0.5, hw.nvme_read_gbps)

    # Total latency per token (overlapped prefetching softens NVMe by ~40%)
    t_memory_sec = t_vram_sec + t_ram_sec + (t_nvme_sec * 0.65)
    token_latency_sec = max(t_compute_sec, t_memory_sec)
    tok_per_sec = round(1.0 / max(0.001, token_latency_sec), 2)

    # Time-To-First-Token (TTFT)
    # Warm prefill for 512 tokens:
    prefill_flops = compute_gflops_per_token * 512
    ttft_warm_sec = round(max(0.3, (prefill_flops / 1000.0) / max(1.0, effective_tflops * 2.5) + (t_ram_sec * 2.0)), 2)
    # Cold start: initial load from NVMe to RAM/VRAM
    ttft_cold_sec = round(total_weight_gb / max(0.8, hw.nvme_read_gbps) + 1.2, 1)

    # Hardware scale multiplier vs physical 4-bit VRAM capacity
    vram_max_4bit_params = (hw.vram_gb - 1.2) / (4.5 / 8.0)  # in Billions
    scale_multiplier_vs_vram = round(model.total_params / max(0.5, vram_max_4bit_params), 2)

    # Bottleneck & Tiering diagnosis
    warnings = []
    if nvme_layers > 0:
        bottleneck = "NVMe-Disk Swap Bound (High PCIe & SSD Latency)"
        memory_tier_status = f"VRAM + RAM + NVMe ({nvme_layers} layers in swap)"
        warnings.append(f"{nvme_layers} layers overflow into NVMe SSD swap. Generation speed is constrained by disk streaming.")
    elif ram_layers > 0:
        bottleneck = "DDR5 System RAM Bandwidth Bound (Optimal Dual-Tier Offload)"
        memory_tier_status = f"VRAM ({vram_layers} layers) + System RAM ({ram_layers} layers)"
    else:
        bottleneck = "VRAM Bandwidth Bound (100% Native GPU Residency - Peak Speed)"
        memory_tier_status = f"100% GPU VRAM ({vram_layers} layers)"

    if ram_free_gb < 1.0 and nvme_layers == 0:
        warnings.append("System RAM is above 95% capacity. Risk of Windows pagefile compression if other apps open.")

    is_supported = (nvme_weight_gb <= hw.nvme_gb) and (tok_per_sec >= 0.1)

    return SimulationResult(
        model=model,
        hardware=hw,
        quantization=quantization,
        context_length=context_length,
        total_weight_gb=round(total_weight_gb, 2),
        active_weight_gb_per_token=round(active_weight_gb_per_token, 2),
        kv_cache_raw_gb=round(kv_cache_raw_gb, 3),
        kv_cache_compressed_gb=round(kv_cache_compressed_gb, 3),
        vram_layers=vram_layers,
        ram_layers=ram_layers,
        nvme_layers=nvme_layers,
        vram_weight_gb=round(vram_weight_gb, 2),
        ram_weight_gb=round(ram_weight_gb, 2),
        nvme_weight_gb=round(nvme_weight_gb, 2),
        vram_used_gb=round(vram_used_gb, 2),
        vram_free_gb=round(vram_free_gb, 2),
        ram_used_gb=round(ram_used_gb, 2),
        ram_free_gb=round(ram_free_gb, 2),
        compute_gflops_per_token=round(compute_gflops_per_token, 2),
        active_transfer_gb_per_token=round(active_weight_gb_per_token, 2),
        token_latency_sec=round(token_latency_sec, 4),
        tok_per_sec=tok_per_sec,
        ttft_warm_sec=ttft_warm_sec,
        ttft_cold_sec=ttft_cold_sec,
        bottleneck=bottleneck,
        memory_tier_status=memory_tier_status,
        scale_multiplier_vs_vram=scale_multiplier_vs_vram,
        is_supported=is_supported,
        warnings=warnings,
    )
