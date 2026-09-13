"""
PHANTOM PLATFORM — Curated Model Catalog
=========================================
A large, categorized catalog of well-known open-weight models (GGUF) that
PHANTOM can install with one command. Browsed from the TUI via `/install`.

Only entry points are used for resolution — the HF repo id. Quantization sizes
are approximate (Q4_K_M reference) and derived for the other quants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _qs(gb_q4: float) -> "Dict[str, float]":
    """Approximate quant sizes (GB) from the Q4_K_M reference size."""
    return {
        "Q2_K": round(gb_q4 * 0.55, 1),
        "Q3_K_M": round(gb_q4 * 0.78, 1),
        "Q4_0": round(gb_q4 * 0.98, 1),
        "Q4_K_M": round(gb_q4, 1),
        "Q5_K_M": round(gb_q4 * 1.20, 1),
        "Q6_K": round(gb_q4 * 1.33, 1),
        "Q8_0": round(gb_q4 * 1.55, 1),
    }


@dataclass(frozen=True)
class CatalogModel:
    id: str                      # short alias shown in the picker
    name: str                    # full human name
    repo: str                    # Hugging Face GGUF repository
    family: str                  # architecture family
    params: str                  # parameter count label
    context: str                 # native context window
    category: str                # catalog category
    q4_gb: float                 # Q4_K_M approx size in GB
    desc: str = ""               # one-line description
    quants: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.quants:
            object.__setattr__(self, "quants", _qs(self.q4_gb))


CATALOG: List[CatalogModel] = [
    # ─────────────────────────────── Code ───────────────────────────────
    CatalogModel("qwen2.5-coder:32b", "Qwen2.5 Coder 32B Instruct", "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
                 "Qwen2.5", "32.5B", "128K", "Code", 19.0, "State-of-the-art open coding LLM — SWE, refactors, multi-file"),
    CatalogModel("qwen2.5-coder:14b", "Qwen2.5 Coder 14B Instruct", "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
                 "Qwen2.5", "14.7B", "128K", "Code", 9.3, "Mid-size coding model — best quality/VRAM sweet spot"),
    CatalogModel("qwen2.5-coder:7b", "Qwen2.5 Coder 7B Instruct", "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
                 "Qwen2.5", "7.6B", "128K", "Code", 4.7, "Fast everyday coding on modest GPUs"),
    CatalogModel("qwen2.5-coder:3b", "Qwen2.5 Coder 3B Instruct", "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF",
                 "Qwen2.5", "3.1B", "128K", "Code", 2.0, "Lightweight coding — 6 GB VRAM friendly"),
    CatalogModel("qwen2.5-coder:1.5b", "Qwen2.5 Coder 1.5B Instruct", "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
                 "Qwen2.5", "1.5B", "128K", "Code", 1.0, "Smallest useful coder — CPU-only capable"),
    CatalogModel("deepseek-coder:33b", "DeepSeek-Coder 33B Instruct", "bartowski/DeepSeek-Coder-33B-Instruct-GGUF",
                 "DeepSeek", "32.8B", "16K", "Code", 20.0, "Strong multilingual coder from DeepSeek"),
    CatalogModel("deepseek-coder:6.7b", "DeepSeek-Coder 6.7B Instruct", "bartowski/deepseek-coder-6.7b-instruct-GGUF",
                 "DeepSeek", "6.7B", "16K", "Code", 4.0, "Compact DeepSeek coder"),
    CatalogModel("codellama:70b", "CodeLlama 70B Instruct", "bartowski/CodeLlama-70b-Instruct-GGUF",
                 "Llama", "69B", "16K", "Code", 40.0, "Meta's largest classic code model"),
    CatalogModel("codellama:34b", "CodeLlama 34B Instruct", "bartowski/CodeLlama-34b-Instruct-GGUF",
                 "Llama", "34B", "16K", "Code", 20.0, "Balanced CodeLlama quality tier"),
    CatalogModel("codellama:13b", "CodeLlama 13B Instruct", "bartowski/CodeLlama-13b-Instruct-GGUF",
                 "Llama", "13B", "16K", "Code", 8.0, "Classic 13B code model"),
    CatalogModel("codellama:7b", "CodeLlama 7B Instruct", "bartowski/CodeLlama-7b-Instruct-GGUF",
                 "Llama", "6.7B", "16K", "Code", 4.3, "Old-but-cheap code model"),
    CatalogModel("starcoder2:15b", "StarCoder2 15B Instruct", "bartowski/StarCoder2-15B-Instruct-v0.1-GGUF",
                 "StarCoder", "15B", "16K", "Code", 9.3, "BigCode's fallback-aware coder"),
    CatalogModel("starcoder2:3b", "StarCoder2 3B Instruct", "bartowski/StarCoder2-3B-Instruct-v0.1-GGUF",
                 "StarCoder", "3B", "16K", "Code", 2.0, "Light BigCode coder"),
    CatalogModel("granite-code:8b", "IBM Granite Code 8B Instruct", "bartowski/ibm-granite-8b-code-instruct-4k-GGUF",
                 "Granite", "8.1B", "4K", "Code", 5.4, "Enterprise-focused code model from IBM"),
    CatalogModel("granite:8b", "IBM Granite 3.1 8B Instruct", "bartowski/granite-3.1-8b-instruct-GGUF",
                 "Granite", "8.1B", "128K", "Code", 5.4, "Apache-2.0 general + code instruct"),
    CatalogModel("granite:3b", "IBM Granite 3.0 3B Instruct", "bartowski/granite-3.0-3b-instruct-GGUF",
                 "Granite", "3.1B", "4K", "Code", 2.0, "Tiny 3B instruct for edge machines"),

    # ─────────────────────────────── General ───────────────────────────────
    CatalogModel("llama3.3:70b", "Llama 3.3 70B Instruct", "bartowski/Llama-3.3-70B-Instruct-GGUF",
                 "Llama", "70.6B", "128K", "General", 40.0, "Meta's flagship open chat model"),
    CatalogModel("llama3.1:8b", "Llama 3.1 8B Instruct", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                 "Llama", "8.0B", "128K", "General", 4.9, "Reliable 8B general chat"),
    CatalogModel("llama3.2:3b", "Llama 3.2 3B Instruct", "bartowski/Llama-3.2-3B-Instruct-GGUF",
                 "Llama", "3.2B", "128K", "General", 2.0, "Speedy 3B for laptops"),
    CatalogModel("llama3.2:1b", "Llama 3.2 1B Instruct", "bartowski/Llama-3.2-1B-Instruct-GGUF",
                 "Llama", "1.2B", "128K", "General", 0.8, "CPU-friendly 1B chat"),
    CatalogModel("qwen2.5:72b", "Qwen2.5 72B Instruct", "bartowski/Qwen2.5-72B-Instruct-GGUF",
                 "Qwen2.5", "72.7B", "128K", "General", 43.0, "Frontier-class open model"),
    CatalogModel("qwen2.5:32b", "Qwen2.5 32B Instruct", "bartowski/Qwen2.5-32B-Instruct-GGUF",
                 "Qwen2.5", "32.8B", "128K", "General", 20.0, "High quality, fits more setups"),
    CatalogModel("qwen2.5:14b", "Qwen2.5 14B Instruct", "bartowski/Qwen2.5-14B-Instruct-GGUF",
                 "Qwen2.5", "14.7B", "128K", "General", 9.3, "Well-rounded 14B chat"),
    CatalogModel("qwen2.5:7b", "Qwen2.5 7B Instruct", "bartowski/Qwen2.5-7B-Instruct-GGUF",
                 "Qwen2.5", "7.6B", "128K", "General", 4.7, "Fast, competent general chat"),
    CatalogModel("qwen2.5:3b", "Qwen2.5 3B Instruct", "bartowski/Qwen2.5-3B-Instruct-GGUF",
                 "Qwen2.5", "3.1B", "128K", "General", 2.0, "Light 3B all-rounder"),
    CatalogModel("mistral:7b", "Mistral 7B Instruct v0.3", "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                 "Mistral", "7.2B", "32K", "General", 4.4, "The classic open 7B"),
    CatalogModel("mistral-nemo:12b", "Mistral NeMo 12B Instruct", "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
                 "Mistral", "12.2B", "128K", "General", 7.5, "Mistral + NVIDIA 12B, very capable"),
    CatalogModel("mixtral:8x7b", "Mixtral 8x7B Instruct v0.1", "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",
                 "Mixtral", "46.7B", "32K", "General", 26.0, "First-gen MoE (disable sparsity routing)"),
    CatalogModel("mistral-small:24b", "Mistral Small 24B Instruct 2501", "bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
                 "Mistral", "24B", "128K", "General", 14.0, "Modern 24B — strong reasoning for the size"),
    CatalogModel("gemma2:27b", "Gemma 2 27B Instruct", "bartowski/gemma-2-27b-it-GGUF",
                 "Gemma", "27B", "8K", "General", 16.0, "Google's high-QoE 27B"),
    CatalogModel("gemma2:9b", "Gemma 2 9B Instruct", "bartowski/gemma-2-9b-it-GGUF",
                 "Gemma", "9.2B", "8K", "General", 5.5, "Google 9B — punchy for its size"),
    CatalogModel("gemma2:2b", "Gemma 2 2B Instruct", "bartowski/gemma-2-2b-it-GGUF",
                 "Gemma", "2.6B", "8K", "General", 1.6, "Tiny capable demo model"),
    CatalogModel("gemma3:27b", "Gemma 3 27B Instruct", "bartowski/gemma-3-27b-it-GGUF",
                 "Gemma", "27.1B", "128K", "General", 16.0, "Latest Google open model"),
    CatalogModel("gemma3:4b", "Gemma 3 4B Instruct", "bartowski/gemma-3-4b-it-GGUF",
                 "Gemma", "4.1B", "128K", "General", 2.7, "New-gen 4B with long context"),
    CatalogModel("phi3.5:3.8b", "Phi-3.5 Mini Instruct", "bartowski/Phi-3.5-mini-instruct-GGUF",
                 "Phi", "3.8B", "128K", "General", 2.2, "Microsoft's excellent small model"),
    CatalogModel("phi4:14b", "Phi-4 14B Instruct", "bartowski/Phi-4-Instruct-GGUF",
                 "Phi", "14.7B", "16K", "General", 9.0, "New-gen Microsoft reasoning-lite model"),
    CatalogModel("yi1.5:34b", "Yi 1.5 34B Chat", "bartowski/Yi-1.5-34B-Chat-GGUF",
                 "Yi", "34.4B", "4K", "General", 20.0, "Bilingual EN/中文 strong model"),
    CatalogModel("yi1.5:9b", "Yi 1.5 9B Chat", "bartowski/Yi-1.5-9B-Chat-GGUF",
                 "Yi", "9.1B", "4K", "General", 5.5, "Bilingual compact chat"),
    CatalogModel("command-r:35b", "Cohere Command R 35B", "bartowski/c4ai-command-r-v01-GGUF",
                 "Command R", "35B", "128K", "General", 21.0, "Enterprise RAG-heavy chat, huge context"),
    CatalogModel("stablelm2:12b", "StableLM 2 12B Zephyr", "bartowski/stablelm-2-12b-GGUF",
                 "StableLM", "12.1B", "4K", "General", 7.4, "Creative-generation tuned"),
    CatalogModel("falcon2:11b", "Falcon2 11B", "bartowski/Falcon2-11B-chat-GGUF",
                 "Falcon", "11.2B", "8K", "General", 7.0, "TII's efficient 11B chat"),
    CatalogModel("dolphin:2.9", "Dolphin 2.9.2 Llama 8B", "bartowski/dolphin-2.9.2-llama-3.1-8b-GGUF",
                 "Llama", "8.0B", "4K", "General", 4.9, "Uncensored fun model (use with judgement)"),

    # ─────────────────────────────── Reasoning ───────────────────────────────
    CatalogModel("deepseek-r1:70b", "DeepSeek-R1 Distill Llama 70B", "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF",
                 "Llama", "70.6B", "128K", "Reasoning", 40.0, "Frontier-class distilled reasoning"),
    CatalogModel("deepseek-r1:32b", "DeepSeek-R1 Distill Qwen 32B", "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
                 "Qwen", "32.8B", "128K", "Reasoning", 19.0, "Best reasoning-per-GB distilled R1"),
    CatalogModel("deepseek-r1:14b", "DeepSeek-R1 Distill Qwen 14B", "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
                 "Qwen", "14.7B", "128K", "Reasoning", 9.3, "Mid reasoning pup"),
    CatalogModel("deepseek-r1:8b", "DeepSeek-R1 Distill Llama 8B", "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
                 "Llama", "8.0B", "128K", "Reasoning", 4.9, "Small fast reasoning"),
    CatalogModel("deepseek-r1:1.5b", "DeepSeek-R1 Distill Qwen 1.5B", "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
                 "Qwen", "1.8B", "128K", "Reasoning", 1.1, "Tiny reasoning — CPU ok"),
    CatalogModel("qwq:32b", "QwQ 32B", "bartowski/QwQ-32B-GGUF",
                 "Qwen", "32.8B", "128K", "Reasoning", 19.0, "Superb open reasoning model"),
    CatalogModel("qwen3:32b", "Qwen3 32B", "bartowski/Qwen3-32B-GGUF",
                 "Qwen3", "32.8B", "128K", "Reasoning", 20.0, "New-gen Qwen thinking mode"),
    CatalogModel("qwen3:30b-a3b", "Qwen3 30B-A3B MoE", "bartowski/Qwen3-30B-A3B-Instruct-GGUF",
                 "Qwen3", "30.5B", "128K", "Reasoning", 18.0, "MoE — fast, ~3B active (disable routing)"),
    CatalogModel("qwen3:14b", "Qwen3 14B", "bartowski/Qwen3-14B-GGUF",
                 "Qwen3", "14.8B", "128K", "Reasoning", 9.5, "Mid-tier Qwen3 thinking"),
    CatalogModel("qwen3:8b", "Qwen3 8B", "bartowski/Qwen3-8B-GGUF",
                 "Qwen3", "8.1B", "128K", "Reasoning", 5.0, "Fast Qwen3"),
    CatalogModel("qwen3:4b", "Qwen3 4B", "bartowski/Qwen3-4B-GGUF",
                 "Qwen3", "3.9B", "128K", "Reasoning", 2.7, "Svelte Qwen3"),
    CatalogModel("openthinker:32b", "OpenThinker 32B", "bartowski/OpenThinker-32B-GGUF",
                 "Qwen", "32.8B", "128K", "Reasoning", 20.0, "Open CoT-heavy reasoning"),

    # ─────────────────────────────── Small & Fast ───────────────────────────────
    CatalogModel("smollm2:135m", "SmolLM2 135M Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF",
                 "SmolLM2", "135M", "8K", "Small & Fast", 0.09, "Instant sanity-check model — install this first"),
    CatalogModel("smollm2:360m", "SmolLM2 360M Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
                 "SmolLM2", "361M", "8K", "Small & Fast", 0.22, "Tiny but coherent"),
    CatalogModel("smollm2:1.7b", "SmolLM2 1.7B Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
                 "SmolLM2", "1.7B", "8K", "Small & Fast", 1.1, "Best tiny-model baseline"),
    CatalogModel("tinyllama:1.1b", "TinyLlama 1.1B Chat", "bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF",
                 "Llama", "1.1B", "2K", "Small & Fast", 0.8, "Degenerate-tester classic"),
    CatalogModel("stablelm2:1.6b", "StableLM 2 1.6B Zephyr", "bartowski/stablelm-2-zephyr-1.6b-GGUF",
                 "StableLM", "1.6B", "4K", "Small & Fast", 1.1, "Mobile-friendly chat"),

    # ─────────────────────────────── Math & Science ───────────────────────────────
    CatalogModel("qwen2.5-math:72b", "Qwen2.5 Math 72B", "bartowski/Qwen2.5-Math-72B-Instruct-GGUF",
                 "Qwen2.5", "72.7B", "128K", "Math & Science", 43.0, "Frontier math specialist"),
    CatalogModel("qwen2.5-math:7b", "Qwen2.5 Math 7B", "bartowski/Qwen2.5-Math-7B-Instruct-GGUF",
                 "Qwen2.5", "7.6B", "128K", "Math & Science", 4.7, "Fast math tutor"),
    CatalogModel("deepseek-math:7b", "DeepSeek-Math 7B RL", "bartowski/deepseek-math-7b-rl-GGUF",
                 "DeepSeek", "7.0B", "4K", "Math & Science", 4.3, "RL-trained math solver"),
    CatalogModel("mathstral:7b", "Mistral Mathstral 7B", "bartowski/Mathstral-7B-v0.1-GGUF",
                 "Mistral", "7.2B", "32K", "Math & Science", 4.4, "Mistral's math-tuned 7B"),
]


CATEGORIES: Tuple[str, ...] = (
    "Code",
    "General",
    "Reasoning",
    "Small & Fast",
    "Math & Science",
)

# Intentional ordering for display; keeps the big list browsable.
_CATEGORY_ORDER = {c: idx for idx, c in enumerate(CATEGORIES)}


def catalog_categories() -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for m in CATALOG:
        counts[m.category] = counts.get(m.category, 0) + 1
    return sorted(counts.items(), key=lambda kv: _CATEGORY_ORDER.get(kv[0], 99))


def catalog_models(category: Optional[str] = None) -> List[CatalogModel]:
    if category:
        return [m for m in CATALOG if m.category == category]
    return list(CATALOG)


def catalog_find(repo_or_id: str) -> Optional[CatalogModel]:
    q = repo_or_id.strip().lower()
    for m in CATALOG:
        if m.id.lower() == q or m.repo.lower() == q:
            return m
    return None