"""
Unit tests for PHANTOM Automated Cloud Testbed & Colab Runner (scripts/colab_runner.py).
Verifies complete 14-model registry, virtual hardware simulation, non-synthetic task battery,
and TEMPLATE_TEST_REPORT.md markdown report generation.
"""

import json
import pytest
from pathlib import Path
from scripts.colab_runner import (
    MODEL_REGISTRY,
    VERIFICATION_TASKS,
    generate_markdown_report,
    run_cloud_benchmark,
)
from phantom.model_profiles.hardware_simulator import simulate_model_execution


EXPECTED_MODELS = [
    "smollm-135m",
    "qwen2.5-coder-32b",
    "qwen3-30b-a3b",
    "llama-3-70b",
    "deepseek-r1-distill-qwen-32b",
    "deepseek-r1-distill-llama-70b",
    "mixtral-8x7b-instruct",
    "qwq-32b-preview",
    "qwen2.5-72b-instruct",
    "qwen2.5-32b-instruct",
    "deepseek-coder-33b-instruct",
    "codellama-70b-instruct",
    "command-r-35b",
    "yi-1.5-34b-chat",
]


def test_model_registry_contains_all_14_models():
    """Verify that all 14 evaluated models are registered with complete metadata."""
    for model_key in EXPECTED_MODELS:
        assert model_key in MODEL_REGISTRY, f"Missing model: {model_key}"
        entry = MODEL_REGISTRY[model_key]
        assert "name" in entry and len(entry["name"]) > 0
        assert "repo" in entry and "/" in entry["repo"]
        assert "file" in entry and entry["file"].endswith(".gguf")
        assert "params_b" in entry and entry["params_b"] > 0
        assert "arch" in entry and len(entry["arch"]) > 0
        assert "quant" in entry
        assert "test_id" in entry and entry["test_id"].startswith("test_")


def test_verification_tasks_definitions():
    """Verify non-synthetic verification tasks structure and ground truth targets."""
    assert len(VERIFICATION_TASKS) == 3
    targets = {t["id"]: t["expected"] for t in VERIFICATION_TASKS}
    assert targets["task_1_knapsack"] == "220"
    assert targets["task_2_harmonic_mean"] == "48"
    assert "reversed" in targets["task_3_word_reversal"]


def test_dry_run_cloud_benchmark(tmp_path: Path):
    """Verify run_cloud_benchmark in dry-run mode outputs valid JSON and Markdown files."""
    model_key = "qwen3-30b-a3b"
    payload, report_md = run_cloud_benchmark(
        model_key=model_key,
        preset="colab-t4",
        dry_run=True,
        output_dir=tmp_path,
    )

    # Check payload structure
    assert payload["model_key"] == model_key
    assert payload["hardware_preset"] == "colab-t4"
    assert payload["dry_run"] is True
    assert "simulation" in payload
    assert payload["simulation"]["tok_per_sec"] > 0
    assert payload["simulation"]["vram_layers"] > 0
    assert len(payload["tasks"]) == 3
    assert all(t["status"] == "PASS" for t in payload["tasks"])

    # Check generated files in tmp_path
    json_files = list(tmp_path.glob("*.json"))
    md_files = list(tmp_path.glob("*.md"))
    assert len(json_files) == 1
    assert len(md_files) == 1

    # Check Markdown report adherence to TEMPLATE_TEST_REPORT.md
    report_content = md_files[0].read_text(encoding="utf-8")
    assert "# Test 03: Qwen3-30B-A3B" in report_content
    assert "## 1. Executive Summary & Hardware Multipliers" in report_content
    assert "## 2. Real Hardware Load & Performance Metrics" in report_content
    assert "## 3. Non-Synthetic Task Verification" in report_content
    assert "## 4. Architectural Summary & Hardware Verdict" in report_content
    assert "Zero-Disk Invariant" in report_content


def test_report_generation_for_frontier_70b(tmp_path: Path):
    """Verify report formatting for dense 3-tier model (Llama-3-70B)."""
    model_key = "llama-3-70b"
    meta = MODEL_REGISTRY[model_key]
    sim = simulate_model_execution(model_key, "rtx4050-laptop")
    out_file = tmp_path / "test_04_llama3_70b.md"

    report = generate_markdown_report(
        model_key=model_key,
        meta=meta,
        sim=sim,
        hw_name="rtx4050-laptop",
        task_results=[{"name": "Knapsack", "prompt": "test", "expected": "220", "status": "PASS"}],
        output_path=out_file,
    )

    assert out_file.exists()
    assert "# Test 04: Llama-3-70B-Instruct" in report
    assert "100% Dense (3-Tier Swap)" in report
    assert f"{sim.nvme_layers}" in report
