#!/usr/bin/env python3
"""
PHANTOM SOP Compliance & Progress Tracking Health Checker
Verifies that all 5 living documentation ledgers are present, active, and properly formatted.
Usage:
    python scripts/verify_tracking.py
"""

import sys
from pathlib import Path

# Force UTF-8 stdout for Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent

TRACKING_FILES = {
    "Daily Workboard": {
        "path": REPO_ROOT / "docs" / "DAILY_WORKBOARD.md",
        "required_strings": ["Active Session Workboard", "Today's Action Checklist", "The Daily 5-Step Operational Loop"],
    },
    "Today Quick Pointer": {
        "path": REPO_ROOT / "TODAY.md",
        "required_strings": ["docs/DAILY_WORKBOARD.md", "Quick Session Summary"],
    },
    "Engineering Changelog": {
        "path": REPO_ROOT / "docs" / "CHANGELOG.md",
        "required_strings": ["PHANTOM Granular Engineering Changelog", "Added"],
    },
    "Progress Dashboard": {
        "path": REPO_ROOT / "docs" / "PROGRESS.md",
        "required_strings": ["PHANTOM Living Progress & Subsystem Health Dashboard", "Subsystem Readiness & Verification Matrix"],
    },
    "Testing Ledger": {
        "path": REPO_ROOT / "docs" / "testing" / "INDEX.md",
        "required_strings": ["PHANTOM Continuous Testing Ledger & Performance Register", "Master Test Run Registry"],
    },
    "Decision Log (ADR)": {
        "path": REPO_ROOT / "docs" / "DECISION_LOG.md",
        "required_strings": ["PHANTOM Architecture Decision Records", "ADR-001"],
    },
    "Agent Rules (AGENTS.md)": {
        "path": REPO_ROOT / "AGENTS.md",
        "required_strings": ["Post-Prompt Mandatory Update Protocol", "docs/CHANGELOG.md"],
    },
    "Workspace SOP (SOP.md)": {
        "path": REPO_ROOT / "docs" / "SOP.md",
        "required_strings": ["Autonomous Progress Tracking", "The 5 Living Documentation Ledgers"],
    },
}


def audit_tracking_ledgers() -> bool:
    print("=" * 72)
    print("  📋 PHANTOM DOCUMENTATION & TRACKING HEALTH AUDIT")
    print("=" * 72)

    all_passed = True
    results = []

    for name, spec in TRACKING_FILES.items():
        file_path = spec["path"]
        rel_path = file_path.relative_to(REPO_ROOT)

        if not file_path.exists():
            results.append((name, str(rel_path), "MISSING", 0, "File not found"))
            all_passed = False
            continue

        size = file_path.stat().st_size
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            results.append((name, str(rel_path), "ERROR", size, f"Encoding error: {e}"))
            all_passed = False
            continue

        missing_reqs = [req for req in spec["required_strings"] if req not in content]
        if missing_reqs:
            results.append((name, str(rel_path), "WARN", size, f"Missing: {missing_reqs[0]}"))
            all_passed = False
        else:
            results.append((name, str(rel_path), "HEALTHY", size, "All markers present"))

    # Print summary table
    print(f"{'Ledger Name':<24} | {'Status':<8} | {'Bytes':<7} | {'Path'}")
    print("-" * 72)
    for name, rel_path, status, size, note in results:
        status_sym = "✓" if status == "HEALTHY" else "✗"
        print(f"{name:<24} | [{status_sym}] {status:<5} | {size:<7} | {rel_path}")

    print("=" * 72)
    if all_passed:
        print("  🎉 [ALL 8 TRACKING LEDGERS HEALTHY & SYNCHRONIZED]")
        print("=" * 72)
        return True
    else:
        print("  ⚠️ [TRACKING AUDIT FAILED — PLEASE UPDATE MISSING LEDGERS]")
        print("=" * 72)
        return False


if __name__ == "__main__":
    success = audit_tracking_ledgers()
    sys.exit(0 if success else 1)
