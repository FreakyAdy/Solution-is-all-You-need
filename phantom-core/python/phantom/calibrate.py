"""
PHANTOM CORE — Full Calibration Pipeline (Innovation Driver)
=============================================================
Master calibration runner. Generates the .phantom profile that unlocks
all 7 PHANTOM CORE innovations for a specific model on your hardware.

Pipeline Steps:
    STEP A — Layer Profiling      (2 min): Activation stats per layer
    STEP B — Spectral K Selection (1 min): DCT coefficient budget per MLP
    STEP C — KV AE Training       (4 min): Neural Cache autoencoder
    STEP D — Sparsity Gate Cal.   (2 min): Adaptive Compute Routing gates
    STEP E — Wraith Warm-up       (1 min): LSTM predictor pre-training

Total target: ≤10 minutes on any hardware tier.

Usage:
    from phantom.calibrate import CalibrationPipeline
    pipeline = CalibrationPipeline("/path/to/model", "/output/dir")
    profile = pipeline.run()
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import structlog

from phantom.loader import load_model_for_calibration
from phantom.model_profiles.auto_detect import detect_architecture
from phantom.model_profiles.hardware_detect import detect_hardware, print_hardware_report
from phantom.spectral_analyzer import analyze_model_spectral
from phantom.neural_cache_ae import (
    KVCollector,
    train_kv_autoencoder,
    export_for_cuda,
    save_pytorch,
)
from phantom.wraith_lstm import WraithPredictor, warm_up_from_log

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration data structures
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationError(Exception):
    """Raised when a calibration step fails or produces invalid results."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Step A: Layer Profiling
# ─────────────────────────────────────────────────────────────────────────────

def _step_a_profile_layers(
    model: nn.Module,
    tokenizer,
    calibration_prompts: List[str],
    num_samples: int = 50,
    device: str = "cuda",
) -> Dict[int, dict]:
    """
    STEP A — Layer Profiling.

    Runs calibration prompts through the model with no optimizations,
    collecting per-layer statistics:
        - activation_l2_mean, activation_l2_var
        - attention_entropy_mean, attention_entropy_var
        - mlp_neuron_activation_freq (fraction of neurons active per token)
        - forward_pass_time_ms

    Args:
        model:               Loaded HuggingFace model.
        tokenizer:           Model tokenizer.
        calibration_prompts: List of diverse prompts.
        num_samples:         Number of prompts to use.
        device:              Computation device.

    Returns:
        Dict mapping layer_id -> {stats dict}.
    """
    logger.info("step_a_start", num_samples=min(num_samples, len(calibration_prompts)))
    t0 = time.time()

    layer_stats: Dict[int, dict] = {}
    activation_records: Dict[int, List[float]] = {}
    entropy_records: Dict[int, List[float]] = {}
    sparsity_records: Dict[int, List[float]] = {}
    timing_records: Dict[int, List[float]] = {}

    # Hook storage
    hooks = []
    hook_data = {"layer_outputs": {}, "attn_weights": {}}

    # Find all transformer layers
    layer_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "self_attn") or (
            hasattr(module, "attention") and hasattr(module, "mlp")
        ):
            layer_modules.append((name, module))

    num_layers = len(layer_modules)
    if num_layers == 0:
        # Fallback: try to detect by naming pattern
        import re
        prev_idx = -1
        for name, mod in model.named_modules():
            m = re.search(r"\.layers?\.(\d+)$", name)
            if m:
                idx = int(m.group(1))
                if idx != prev_idx:
                    layer_modules.append((name, mod))
                    prev_idx = idx
        num_layers = len(layer_modules)

    # Register output hooks on each layer
    def make_hook(layer_idx: int):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                out = outputs[0]
            else:
                out = outputs
            if isinstance(out, torch.Tensor):
                l2 = out.detach().float().norm(dim=-1).mean().item()
                if layer_idx not in activation_records:
                    activation_records[layer_idx] = []
                activation_records[layer_idx].append(l2)

                # Sparsity: fraction of near-zero activations
                sparsity = (out.detach().abs() < 0.01 * out.detach().abs().max()).float().mean().item()
                if layer_idx not in sparsity_records:
                    sparsity_records[layer_idx] = []
                sparsity_records[layer_idx].append(sparsity)
        return hook

    for i, (name, module) in enumerate(layer_modules):
        h = module.register_forward_hook(make_hook(i))
        hooks.append(h)

    model.eval()
    prompts_used = calibration_prompts[:num_samples]

    for prompt_idx, prompt in enumerate(prompts_used):
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            ).to(device)

            with torch.no_grad():
                t_fwd = time.time()
                outputs = model(**inputs, output_attentions=True)
                fwd_ms = (time.time() - t_fwd) * 1000

            # Collect attention entropy from outputs
            if hasattr(outputs, "attentions") and outputs.attentions:
                for layer_idx, attn_weights in enumerate(outputs.attentions):
                    if attn_weights is None:
                        continue
                    # Compute entropy of attention distribution
                    probs = attn_weights.detach().float().clamp(min=1e-9)
                    entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
                    if layer_idx not in entropy_records:
                        entropy_records[layer_idx] = []
                    entropy_records[layer_idx].append(entropy)

                    if layer_idx not in timing_records:
                        timing_records[layer_idx] = []
                    timing_records[layer_idx].append(fwd_ms / max(num_layers, 1))

        except Exception as e:
            logger.warning("step_a_sample_failed", prompt_idx=prompt_idx, error=str(e))
            continue

        if (prompt_idx + 1) % 10 == 0:
            logger.info("step_a_progress", done=prompt_idx + 1, total=len(prompts_used))

    # Remove hooks
    for h in hooks:
        h.remove()

    # Aggregate stats
    for layer_id in range(num_layers):
        l2_vals = activation_records.get(layer_id, [1.0])
        ent_vals = entropy_records.get(layer_id, [1.0])
        spa_vals = sparsity_records.get(layer_id, [0.5])
        tim_vals = timing_records.get(layer_id, [1.0])

        layer_stats[layer_id] = {
            "activation_l2_mean": float(np.mean(l2_vals)),
            "activation_l2_var": float(np.var(l2_vals)),
            "attention_entropy_mean": float(np.mean(ent_vals)),
            "attention_entropy_var": float(np.var(ent_vals)),
            "mlp_neuron_activation_freq": float(1.0 - np.mean(spa_vals)),
            "forward_pass_time_ms": float(np.mean(tim_vals)),
        }

    elapsed = time.time() - t0
    logger.info(
        "step_a_done",
        num_layers_profiled=len(layer_stats),
        elapsed_sec=round(elapsed, 1),
    )
    return layer_stats


# ─────────────────────────────────────────────────────────────────────────────
# Step D: Sparsity Gate Calibration
# ─────────────────────────────────────────────────────────────────────────────

def _step_d_calibrate_gates(
    model: nn.Module,
    tokenizer,
    calibration_prompts: List[str],
    num_samples: int = 30,
    device: str = "cuda",
    activation_threshold_frac: float = 0.01,
    min_f1: float = 0.70,
) -> Tuple[Dict[int, dict], Dict[str, np.ndarray]]:
    """
    STEP D — Sparsity Gate Calibration.

    For each MLP block, trains a linear gate to predict neuron activation masks.
    gate = sigmoid(W_gate @ x + b_gate)

    Training:
        - Loss: Binary cross-entropy
        - Target: ≥85% precision on predicting active neurons
        - Fallback: disable sparse routing for F1 < 0.70 layers

    Args:
        model:               Loaded model.
        tokenizer:           Model tokenizer.
        calibration_prompts: Calibration prompts.
        num_samples:         Number of prompts.
        device:              Computation device.
        activation_threshold_frac: Threshold for "active" neuron (fraction of max).
        min_f1:              Minimum F1 to enable sparse routing.

    Returns:
        Tuple of:
            gate_config: {layer_id: {"enabled": bool, "threshold": float, "precision": float}}
            gate_weights: {layer_id_str: np.ndarray of gate weight matrix}
    """
    logger.info("step_d_start", num_samples=min(num_samples, len(calibration_prompts)))
    t0 = time.time()

    # Collect MLP inputs and activation masks
    mlp_data: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

    # Find MLP gate/up/down layers
    mlp_layer_idx = 0
    mlp_hooks = []
    mlp_hook_store: Dict[int, List] = {}

    def make_mlp_hook(layer_idx: int):
        def hook(module, inputs, outputs):
            if layer_idx not in mlp_hook_store:
                mlp_hook_store[layer_idx] = []
            inp = inputs[0].detach()  # (B, T, D_in)
            out = outputs.detach()    # (B, T, D_out)
            mlp_hook_store[layer_idx].append((
                inp.reshape(-1, inp.shape[-1]).cpu(),
                out.reshape(-1, out.shape[-1]).cpu(),
            ))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            kw in name.lower() for kw in ["gate_proj", "up_proj", "fc1"]
        ):
            h = module.register_forward_hook(make_mlp_hook(mlp_layer_idx))
            mlp_hooks.append(h)
            mlp_layer_idx += 1

    model.eval()
    for prompt in calibration_prompts[:num_samples]:
        try:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            with torch.no_grad():
                model(**inputs)
        except Exception as e:
            logger.warning("step_d_sample_failed", error=str(e))

    for h in mlp_hooks:
        h.remove()

    # Train linear gates for each MLP layer
    gate_config: Dict[int, dict] = {}
    gate_weights: Dict[str, np.ndarray] = {}

    for layer_id, samples in mlp_hook_store.items():
        if not samples:
            gate_config[layer_id] = {"enabled": False, "threshold": 0.5, "precision": 0.0}
            continue

        # Collect all inputs and outputs
        all_inputs = torch.cat([s[0] for s in samples], dim=0).float()
        all_outputs = torch.cat([s[1] for s in samples], dim=0).float()

        # Create binary activation mask
        max_act = all_outputs.abs().max(dim=0, keepdim=True).values.clamp(min=1e-6)
        mask = (all_outputs.abs() >= activation_threshold_frac * max_act).float()

        n, d_in = all_inputs.shape
        d_out = all_outputs.shape[1]

        if n < 10 or d_out > 65536:  # Skip if too few samples or too large
            gate_config[layer_id] = {"enabled": False, "threshold": 0.5, "precision": 0.0}
            continue

        # Train linear gate: W_gate (d_out, d_in), b_gate (d_out)
        W_gate = nn.Parameter(torch.randn(d_out, d_in) * 0.01)
        b_gate = nn.Parameter(torch.zeros(d_out))
        optimizer = torch.optim.AdamW([W_gate, b_gate], lr=1e-3)
        bce = nn.BCEWithLogitsLoss()

        # Mini training loop
        n_epochs = 10
        batch_size = min(256, n)
        for epoch in range(n_epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                x_b = all_inputs[idx]
                y_b = mask[idx]
                logits = x_b @ W_gate.T + b_gate
                loss = bce(logits, y_b)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([W_gate, b_gate], 1.0)
                optimizer.step()
                epoch_loss += loss.item()

        # Evaluate precision, recall, F1
        with torch.no_grad():
            logits = all_inputs @ W_gate.T + b_gate
            preds = (torch.sigmoid(logits) > 0.5).float()
            tp = (preds * mask).sum().item()
            fp = (preds * (1 - mask)).sum().item()
            fn = ((1 - preds) * mask).sum().item()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        enabled = f1 >= min_f1
        gate_config[layer_id] = {
            "enabled": enabled,
            "threshold": 0.5,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

        if enabled:
            gate_weights[str(layer_id)] = {
                "W": W_gate.detach().cpu().numpy().astype(np.float16),
                "b": b_gate.detach().cpu().numpy().astype(np.float16),
            }

        logger.debug(
            "gate_calibrated",
            layer_id=layer_id,
            enabled=enabled,
            f1=round(f1, 3),
            precision=round(precision, 3),
        )

    enabled_count = sum(1 for v in gate_config.values() if v.get("enabled", False))
    elapsed = time.time() - t0
    logger.info(
        "step_d_done",
        total_layers=len(gate_config),
        enabled_layers=enabled_count,
        elapsed_sec=round(elapsed, 1),
    )

    return gate_config, gate_weights


# ─────────────────────────────────────────────────────────────────────────────
# Main calibration pipeline class
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationPipeline:
    """
    PHANTOM CORE Calibration Pipeline.

    Runs all 5 calibration steps for a model and saves the unified
    .phantom profile to the output directory.

    Args:
        model_path:   Path to model (HF directory, GGUF, safetensors).
        output_dir:   Directory to save the .phantom profile.
        num_samples:  Number of calibration prompts (default 50).
        device:       Computation device ("cuda", "cpu", "auto").
        prompts_path: Path to calibration prompts JSONL file.
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        num_samples: int = 50,
        device: str = "auto",
        prompts_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.prompts_path = prompts_path

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "calibration_init",
            model_path=model_path,
            output_dir=str(self.output_dir),
            device=self.device,
        )

    def _load_prompts(self) -> List[str]:
        """Load calibration prompts from JSONL file or use built-in defaults."""
        if self.prompts_path:
            p = Path(self.prompts_path)
        else:
            # Default to bundled calibration prompts
            p = Path(__file__).parent.parent.parent / "calibration" / "datasets" / "calibration_prompts.jsonl"

        if p.exists():
            prompts = []
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            prompts.append(obj.get("text", obj.get("prompt", str(obj))))
                        else:
                            prompts.append(str(obj))
                    except Exception:
                        prompts.append(line)
            if prompts:
                logger.info("prompts_loaded", count=len(prompts), path=str(p))
                return prompts

        # Built-in fallback prompts (diverse)
        logger.warning("using_fallback_prompts")
        return _get_default_prompts()

    def run(self) -> dict:
        """
        Execute the full 5-step calibration pipeline.

        Returns:
            The complete phantom profile dict (also saved to disk).
        """
        total_start = time.time()
        logger.info("calibration_pipeline_start", model=self.model_path)
        print(f"\n[PHANTOM CORE] Starting calibration for: {self.model_path}")
        print(f"[PHANTOM CORE] Output: {self.output_dir}")
        print(f"[PHANTOM CORE] Device: {self.device}\n")

        # ─── Hardware detection ───────────────────────────────────────────
        print("[PHANTOM CORE] Detecting hardware...")
        try:
            hw_profile = detect_hardware()
            print_hardware_report(hw_profile)
            hw_tier = hw_profile.tier
        except Exception as e:
            logger.warning("hardware_detection_failed", error=str(e))
            hw_tier = "UNKNOWN"
            hw_profile = None

        # ─── Model loading ────────────────────────────────────────────────
        print("[PHANTOM CORE] Loading model for calibration...")
        model, tokenizer = load_model_for_calibration(
            self.model_path,
            device=self.device,
            load_in_4bit=(self.device == "cuda" and hw_tier in ("LAPTOP", "MID")),
        )

        # Detect architecture
        try:
            arch = detect_architecture(self.model_path)
            model_config = {
                "name": arch.model_name,
                "num_layers": arch.num_layers,
                "hidden_dim": arch.hidden_dim,
                "num_heads": arch.num_heads,
                "num_kv_heads": arch.num_kv_heads,
                "ffn_dim": arch.ffn_dim,
                "head_dim": arch.head_dim,
                "vocab_size": arch.vocab_size,
                "max_seq_len": arch.max_seq_len,
                "rope_base": arch.rope_base,
                "arch_type": arch.arch_type,
                "total_params": arch.total_params,
                "layer_size_bytes": arch.total_bytes_fp16 // max(arch.num_layers, 1),
            }
        except Exception as e:
            logger.warning("arch_detect_failed", error=str(e))
            # Derive from model itself
            n_params = sum(p.numel() for p in model.parameters())
            n_layers = getattr(model.config, "num_hidden_layers", 32)
            h_dim = getattr(model.config, "hidden_size", 4096)
            model_config = {
                "name": self.model_path,
                "num_layers": n_layers,
                "hidden_dim": h_dim,
                "num_heads": getattr(model.config, "num_attention_heads", 32),
                "num_kv_heads": getattr(model.config, "num_key_value_heads",
                                       getattr(model.config, "num_attention_heads", 32)),
                "ffn_dim": getattr(model.config, "intermediate_size", h_dim * 4),
                "head_dim": h_dim // max(getattr(model.config, "num_attention_heads", 32), 1),
                "vocab_size": getattr(model.config, "vocab_size", 32000),
                "max_seq_len": getattr(model.config, "max_position_embeddings", 4096),
                "rope_base": getattr(model.config, "rope_theta", 10000.0),
                "arch_type": getattr(model.config, "model_type", "llama"),
                "total_params": n_params,
                "layer_size_bytes": (n_params * 2) // max(n_layers, 1),
            }

        # Model hash
        model_hash = hashlib.sha256(
            f"{model_config['name']}-{model_config['total_params']}".encode()
        ).hexdigest()[:16]

        # Load calibration prompts
        prompts = self._load_prompts()
        step_prompts = prompts[:self.num_samples]

        # ─── STEP A: Layer Profiling ──────────────────────────────────────
        print(f"\n[STEP A/5] Layer Profiling... (est. 2 min)")
        layer_stats = _step_a_profile_layers(
            model, tokenizer, step_prompts,
            num_samples=self.num_samples,
            device=self.device,
        )
        print(f"  ✓ Profiled {len(layer_stats)} layers")

        # ─── STEP B: Spectral K Selection ────────────────────────────────
        print(f"\n[STEP B/5] Spectral K Selection... (est. 1 min)")
        try:
            spectral_k_map = analyze_model_spectral(
                model=model,
                tokenizer=tokenizer,
                calibration_texts=step_prompts[:20],
                use_fisher=True,
                device=self.device,
            )
            # Convert int keys to str for JSON compatibility, then back
            spectral_k_map_json = {str(k): int(v) for k, v in spectral_k_map.items()}
        except Exception as e:
            logger.warning("step_b_failed", error=str(e), fallback="default_k")
            n_layers = model_config["num_layers"]
            k_default = int(model_config["ffn_dim"] * 0.15)
            spectral_k_map_json = {str(i): k_default for i in range(n_layers * 3)}

        print(f"  ✓ Computed K for {len(spectral_k_map_json)} MLP weight matrices")

        # ─── STEP C: KV Autoencoder Training ─────────────────────────────
        print(f"\n[STEP C/5] KV Autoencoder Training... (est. 4 min)")
        kv_ae_path = self.output_dir / "kv_autoencoder.pt"
        kv_ae_cuda_path = self.output_dir / "kv_ae_cuda_weights.bin"

        try:
            collector = KVCollector(max_samples=50_000)
            collector.register(model)

            for prompt in step_prompts[:20]:
                try:
                    inputs = tokenizer(
                        prompt, return_tensors="pt", truncation=True, max_length=256
                    ).to(self.device)
                    with torch.no_grad():
                        model(**inputs)
                except Exception:
                    pass

            collector.remove()
            kv_samples = collector.get_samples()

            head_dim = model_config["head_dim"]
            if kv_samples.shape[1] != head_dim:
                # Reshape to match head_dim
                if kv_samples.shape[1] % head_dim == 0:
                    kv_samples = kv_samples.reshape(-1, head_dim)
                else:
                    kv_samples = kv_samples[:, :head_dim]

            kv_model = train_kv_autoencoder(
                kv_samples=kv_samples,
                head_dim=head_dim,
                num_epochs=20,
                device=self.device,
            )
            save_pytorch(kv_model, kv_ae_path)
            export_for_cuda(kv_model, kv_ae_cuda_path)
            print(f"  ✓ KV autoencoder trained. {head_dim}→{head_dim//8} compression")

        except Exception as e:
            logger.warning("step_c_failed", error=str(e), fallback="empty_ae")
            kv_ae_path.write_bytes(b"KV_AE_PLACEHOLDER")
            kv_ae_cuda_path.write_bytes(b"KV_AE_CUDA_PLACEHOLDER")
            print(f"  ⚠ KV AE training failed ({e}). Using placeholder.")

        # ─── STEP D: Sparsity Gate Calibration ───────────────────────────
        print(f"\n[STEP D/5] Sparsity Gate Calibration... (est. 2 min)")
        gate_weights_path = self.output_dir / "gate_weights.npz"
        gate_config_path = self.output_dir / "gate_config.json"

        try:
            gate_config, gate_weights = _step_d_calibrate_gates(
                model=model,
                tokenizer=tokenizer,
                calibration_prompts=step_prompts[:30],
                num_samples=30,
                device=self.device,
            )

            # Save gate weights
            if gate_weights:
                np.savez_compressed(
                    gate_weights_path,
                    **{f"layer_{k}_W": v["W"] for k, v in gate_weights.items()},
                    **{f"layer_{k}_b": v["b"] for k, v in gate_weights.items()},
                )
            else:
                np.savez_compressed(gate_weights_path)

            with open(gate_config_path, "w") as f:
                json.dump(gate_config, f, indent=2)

            enabled = sum(1 for v in gate_config.values() if v.get("enabled", False))
            print(f"  ✓ Gates calibrated. {enabled}/{len(gate_config)} layers have sparse routing enabled")

        except Exception as e:
            logger.warning("step_d_failed", error=str(e), fallback="default_gates")
            n_layers = model_config["num_layers"]
            gate_config = {str(i): {"enabled": True, "threshold": 0.5} for i in range(n_layers)}
            gate_weights_path.write_bytes(b"GATES_PLACEHOLDER")
            gate_config_path.write_text(json.dumps(gate_config))
            print(f"  ⚠ Gate calibration failed ({e}). Using defaults.")

        # ─── STEP E: Wraith Warm-up ───────────────────────────────────────
        print(f"\n[STEP E/5] Wraith LSTM Warm-up... (est. 1 min)")
        wraith_path = self.output_dir / "wraith_init.pt"

        try:
            num_layers = model_config["num_layers"]
            predictor = WraithPredictor(num_layers=num_layers)

            # Generate access log from layer statistics
            access_log = []
            for i, layer_id in enumerate(range(num_layers)):
                stats = layer_stats.get(layer_id, {})
                access_log.append({
                    "layer_id": layer_id,
                    "attn_entropy": stats.get("attention_entropy_mean", 1.0),
                    "l2_norm": stats.get("activation_l2_mean", 1.0),
                    "tok_pos": i * 10,
                })
            # Repeat in forward order (typical access pattern)
            access_log = access_log * (30 // max(len(access_log) // num_layers, 1) + 1)

            warm_up_from_log(predictor, access_log, num_epochs=5)
            predictor.save(wraith_path)
            print(f"  ✓ Wraith LSTM pre-trained with {len(access_log)} access observations")

        except Exception as e:
            logger.warning("step_e_failed", error=str(e), fallback="empty_wraith")
            wraith_path.write_bytes(b"WRAITH_PLACEHOLDER")
            print(f"  ⚠ Wraith warm-up failed ({e}). Using cold-start predictor.")

        # ─── Build unified .phantom profile ──────────────────────────────
        print(f"\n[PHANTOM CORE] Building unified profile...")

        # Average sparsity from layer stats
        avg_sparsity = {
            str(lid): round(1.0 - stats.get("mlp_neuron_activation_freq", 0.5), 3)
            for lid, stats in layer_stats.items()
        }

        profile = {
            "model_hash": model_hash,
            "model_config": model_config,
            "spectral_k_map": spectral_k_map_json,
            "gate_config": {str(k): v for k, v in gate_config.items()},
            "avg_sparsity": avg_sparsity,
            "kv_autoencoder_path": str(kv_ae_path),
            "gate_weights_path": str(gate_weights_path),
            "wraith_init_path": str(wraith_path),
            "hardware_tier": hw_tier,
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "layer_stats": {str(k): v for k, v in layer_stats.items()},
        }

        # Save profile
        profile_path = self.output_dir / f"{model_hash}.phantom"
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)

        total_elapsed = time.time() - total_start
        print(f"\n{'='*62}")
        print(f"[PHANTOM CORE] Calibration COMPLETE!")
        print(f"  Profile: {profile_path}")
        print(f"  Total time: {total_elapsed/60:.1f} min")
        print(f"  Model hash: {model_hash}")
        print(f"{'='*62}\n")

        logger.info(
            "calibration_complete",
            profile_path=str(profile_path),
            elapsed_min=round(total_elapsed / 60, 1),
            model_hash=model_hash,
        )

        return profile


# ─────────────────────────────────────────────────────────────────────────────
# Built-in fallback prompts
# ─────────────────────────────────────────────────────────────────────────────

def _get_default_prompts() -> List[str]:
    """Return a diverse set of default calibration prompts."""
    return [
        # Technical / Code
        "Write a Python function that implements quicksort algorithm with detailed comments.",
        "Explain the difference between TCP and UDP protocols in networking.",
        "How does garbage collection work in Java? Explain generational GC.",
        "Write a Rust function to parse JSON without external crates.",
        "Describe the SOLID principles of object-oriented design with examples.",
        "What is the time complexity of Dijkstra's algorithm and why?",
        "Explain how transformers work in machine learning, including attention.",
        "Write a SQL query to find the second highest salary in an employee table.",
        "How does the Linux kernel manage memory with virtual address spaces?",
        "Explain CUDA programming model: threads, blocks, and grids.",

        # Mathematics / Reasoning
        "Prove that the square root of 2 is irrational.",
        "Solve: If a train travels 120 km/h and another 80 km/h toward each other from 500 km apart, when do they meet?",
        "Explain Bayes' theorem and give a real-world example of its application.",
        "What is the Riemann hypothesis and why does it matter?",
        "Derive the formula for the sum of an arithmetic series from first principles.",
        "How do you calculate the eigenvalues of a 3x3 matrix?",
        "Explain gradient descent and backpropagation in neural networks.",
        "What is a Fourier transform and what is it used for?",

        # Factual / Knowledge
        "What caused World War I? Describe the main contributing factors.",
        "Explain the structure of DNA and how genetic information is encoded.",
        "How does the human immune system respond to a bacterial infection?",
        "Describe the process of photosynthesis at the molecular level.",
        "What are black holes and how do we detect them?",
        "Explain quantum entanglement and the EPR paradox.",
        "How does CRISPR-Cas9 gene editing work?",
        "Describe the economic theory of comparative advantage.",

        # Creative Writing
        "Write a short story about an astronaut who discovers life on Europa.",
        "Compose a poem about artificial intelligence and consciousness.",
        "Write a news article from the year 2150 about climate change.",
        "Create a dialogue between a philosopher and an AI about free will.",

        # Conversational
        "What are the most important skills to develop for a career in software engineering?",
        "How should I approach learning a new programming language efficiently?",
        "What are the pros and cons of remote work versus office work?",
        "Recommend a structured approach to learning machine learning from scratch.",
        "What books would you recommend for someone interested in cognitive science?",

        # Multilingual (diverse distribution)
        "Explain the concept of democracy in simple terms.",
        "What is the relationship between culture and language?",
        "How do you learn to think critically about information you encounter?",

        # Long-form
        "Write a comprehensive guide to setting up a home network, including security best practices.",
        "Explain how modern browsers render web pages from HTML to pixels on screen.",
        "Describe the history of artificial intelligence from Alan Turing to GPT-4.",
        "Write a detailed tutorial on using Git for version control in a team environment.",

        # Edge cases (short inputs)
        "Hello!",
        "What is 2+2?",
        "Translate 'Good morning' to Spanish.",
        "Define entropy.",
        "Name the planets in our solar system.",
        "What year did the Berlin Wall fall?",
        "Who invented the telephone?",
        "What is the speed of light?",
    ]
