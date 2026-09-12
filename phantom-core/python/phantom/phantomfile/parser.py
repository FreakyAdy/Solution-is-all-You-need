"""
PHANTOM PLATFORM — Phantomfile Parser
======================================
Parses declarative Phantomfile specifications and Ollama Modelfiles
into validated model execution configurations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from phantom.phantomfile.validator import PhantomfileValidator, ValidationError


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class PhantomfileConfig:
    base_model: str = ""
    system_prompt: str = ""
    template: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    phantom_params: Dict[str, Any] = field(default_factory=dict)
    plugins: List[str] = field(default_factory=list)
    license: Optional[str] = None
    messages: List[ConversationTurn] = field(default_factory=list)
    stop_tokens: List[str] = field(default_factory=list)


class PhantomfileParser:
    """Parses and validates Phantomfiles."""

    def parse_string(self, content: str) -> PhantomfileConfig:
        config = PhantomfileConfig()
        lines = content.splitlines()
        i = 0
        n = len(lines)

        while i < n:
            raw_line = lines[i].strip()
            i += 1

            # Skip comments and empty lines
            if not raw_line or raw_line.startswith("#"):
                continue

            # Split directive and payload
            parts = raw_line.split(maxsplit=1)
            directive = parts[0].upper()
            payload = parts[1].strip() if len(parts) > 1 else ""

            if directive == "FROM":
                config.base_model = payload.strip("'\"")

            elif directive == "SYSTEM":
                # Check for triple-quote multi-line system prompt
                if payload.startswith('"""') or payload.startswith("'''"):
                    quote_char = payload[:3]
                    content_parts = [payload[3:]]
                    if payload.endswith(quote_char) and len(payload) > 6:
                        # Single-line triple quote
                        config.system_prompt = payload[3:-3].strip()
                    else:
                        while i < n:
                            sub_line = lines[i]
                            i += 1
                            if quote_char in sub_line:
                                content_parts.append(sub_line.split(quote_char)[0])
                                break
                            content_parts.append(sub_line)
                        config.system_prompt = "\n".join(content_parts).strip()
                else:
                    config.system_prompt = payload.strip("'\"")

            elif directive == "TEMPLATE":
                config.template = payload

            elif directive == "PARAMETER":
                p_parts = payload.split(maxsplit=1)
                if len(p_parts) == 2:
                    p_name, p_val = p_parts[0].lower(), p_parts[1].strip("'\"")
                    if p_name == "stop":
                        config.stop_tokens.append(p_val)
                    else:
                        # Parse typed numbers
                        try:
                            if "." in p_val:
                                config.parameters[p_name] = float(p_val)
                            else:
                                config.parameters[p_name] = int(p_val)
                        except ValueError:
                            config.parameters[p_name] = p_val

            elif directive == "PHANTOM_PARAM":
                p_parts = payload.split(maxsplit=1)
                if len(p_parts) == 2:
                    k, v = p_parts[0].lower(), p_parts[1].lower().strip("'\"")
                    if v in ("on", "true", "yes"):
                        config.phantom_params[k] = "on"
                    elif v in ("off", "false", "no"):
                        config.phantom_params[k] = "off"
                    else:
                        try:
                            config.phantom_params[k] = int(v)
                        except ValueError:
                            config.phantom_params[k] = v

            elif directive == "PLUGIN":
                plugin_name = payload.strip()
                if plugin_name not in config.plugins:
                    config.plugins.append(plugin_name)

            elif directive == "LICENSE":
                config.license = payload.strip("'\"")

            elif directive == "MESSAGE":
                m_parts = payload.split(maxsplit=1)
                if len(m_parts) == 2:
                    role = m_parts[0].lower()
                    text = m_parts[1].strip("'\"")
                    config.messages.append(ConversationTurn(role=role, content=text))

        return config

    def parse_file(self, file_path: Union[str, Path]) -> PhantomfileConfig:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Phantomfile not found at {p}")
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        return self.parse_string(content)

    def validate(self, config: PhantomfileConfig) -> List[ValidationError]:
        return PhantomfileValidator.validate(config)

    def to_modelcard(self, config: PhantomfileConfig) -> str:
        """Generate human-readable model card."""
        lines = [
            f"# Model Card: {config.base_model} (Custom Configuration)",
            "",
            "## Architecture & Base Model",
            f"- **Base Model**: `{config.base_model}`",
            f"- **License**: {config.license or 'Inherited from base model'}",
            "",
            "## System Prompt",
            f"> {config.system_prompt or '(Default system prompt)'}",
            "",
            "## Sampling Parameters",
        ]
        for k, v in config.parameters.items():
            lines.append(f"- **{k}**: `{v}`")
        if config.stop_tokens:
            lines.append(f"- **stop tokens**: `{config.stop_tokens}`")

        lines.extend([
            "",
            "## PHANTOM Core Innovations Enabled",
        ])
        for k, v in config.phantom_params.items():
            lines.append(f"- **{k}**: `{v}`")

        if config.plugins:
            lines.extend([
                "",
                "## Active Plugins",
            ])
            for pl in config.plugins:
                lines.append(f"- 🔌 `{pl}`")

        return "\n".join(lines)
