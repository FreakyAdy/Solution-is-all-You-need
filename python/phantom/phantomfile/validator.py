"""
PHANTOM PLATFORM — Phantomfile Validator
=========================================
Validates parsed Phantomfile directives against physical and logical constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValidationError:
    directive: str
    message: str
    line_number: Optional[int] = None

    def __str__(self) -> str:
        loc = f" (line {self.line_number})" if self.line_number else ""
        return f"[{self.directive}]{loc} {self.message}"


class PhantomfileValidator:
    """Validates parameters and architectural constraints for Phantomfile."""

    @staticmethod
    def validate(config: Any) -> List[ValidationError]:
        errors: List[ValidationError] = []

        if not config.base_model:
            errors.append(ValidationError("FROM", "Base model reference 'FROM <model>' is required."))

        # Parameter validation
        params = config.parameters
        if "temperature" in params:
            t = params["temperature"]
            if not isinstance(t, (int, float)) or t < 0.0 or t > 2.0:
                errors.append(ValidationError("PARAMETER temperature", f"Temperature must be between 0.0 and 2.0, got {t}"))

        if "top_p" in params:
            tp = params["top_p"]
            if not isinstance(tp, (int, float)) or tp < 0.0 or tp > 1.0:
                errors.append(ValidationError("PARAMETER top_p", f"top_p must be between 0.0 and 1.0, got {tp}"))

        if "top_k" in params:
            tk = params["top_k"]
            if not isinstance(tk, int) or tk < 1:
                errors.append(ValidationError("PARAMETER top_k", f"top_k must be a positive integer, got {tk}"))

        if "context_window" in params:
            cw = params["context_window"]
            if not isinstance(cw, int) or cw < 128 or cw > 1048576:
                errors.append(ValidationError("PARAMETER context_window", f"context_window out of supported bounds [128, 1M], got {cw}"))

        # PHANTOM_PARAM validation
        phantom_params = config.phantom_params
        for flag in ["sparsity_routing", "kv_compression", "spectral_quant", "safe_mode"]:
            if flag in phantom_params:
                val = phantom_params[flag]
                if val not in ("on", "off", True, False):
                    errors.append(ValidationError(f"PHANTOM_PARAM {flag}", f"Flag must be 'on' or 'off', got '{val}'"))

        if "tier_preference" in phantom_params:
            tp = phantom_params["tier_preference"]
            if tp not in ("vram", "ram", "nvme"):
                errors.append(ValidationError("PHANTOM_PARAM tier_preference", f"Must be 'vram', 'ram', or 'nvme', got '{tp}'"))

        if "max_vram_mb" in phantom_params:
            mv = phantom_params["max_vram_mb"]
            if not isinstance(mv, int) or mv < 512:
                errors.append(ValidationError("PHANTOM_PARAM max_vram_mb", f"max_vram_mb must be integer >= 512 MB, got {mv}"))

        return errors
