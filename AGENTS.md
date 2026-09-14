# AGENTS.md — PHANTOM Agent Rules & Standard Operating Procedure (SOP)

> **MANDATORY DIRECTIVE FOR ALL AGENTS WORKING IN THIS REPOSITORY**:
> Every prompt completion MUST conclude with an evaluation of progress and an update to the corresponding tracking markdown files. Never yield back to the user without verifying and synchronizing these files.

---

## 🔄 Post-Prompt Mandatory Update Protocol (The 5-Ledger SOP)

At the conclusion of EVERY prompt, before delivering your final response to the user, evaluate the actions taken during your turn against this checklist and update the relevant files:

### 1. Daily Workboard & Active Session (`TODAY.md` & `docs/DAILY_WORKBOARD.md`)
- **When to update**: **ALWAYS.** Every single prompt turn.
- **What to do**:
  - Check off any completed task items (`[x]`).
  - Update the "Current Focus" or add new emergent action items.
  - Keep the "Next Session Queue" fresh and ready for tomorrow or the next turn.

### 2. Engineering Changelog (`docs/CHANGELOG.md`)
- **When to update**: Whenever any code, test, configuration, script, or documentation file was created, modified, or deleted.
- **What to do**:
  - Add a concise bullet under `## [Unreleased]` (or the current session commit header) categorized by `Added`, `Changed`, `Fixed`, or `Optimized`.
  - Reference the exact relative file paths and functions touched.

### 3. Subsystem Progress Scorecard (`docs/PROGRESS.md`)
- **When to update**: Whenever a subsystem milestone is reached, a bug is fixed, a new test passes, or readiness metrics change.
- **What to do**:
  - Update the subsystem completion percentage or status indicator (e.g. `100%`, `IN PROGRESS`, `VERIFIED`).
  - Update the "Recent Milestone Chronology" table.

### 4. Testing & Benchmark Ledger (`docs/testing/INDEX.md`)
- **When to update**: Whenever any test suite, unit test, simulation, or model inference test was executed.
- **What to do**:
  - If a model was tested (e.g. via `tests/ephemeral_test_runner.py` or manual run), verify that a row exists in `docs/testing/INDEX.md`.
  - If a comprehensive benchmark was run, create a dedicated report (`docs/testing/test_XX_<model>.md`) following [`docs/testing/TEMPLATE_TEST_REPORT.md`](file:///c:/Work/Projects/Solution%20is%20all%20You%20need/docs/testing/TEMPLATE_TEST_REPORT.md).

### 5. Architecture Decision Records (`docs/DECISION_LOG.md`)
- **When to update**: Whenever an architectural trade-off, hardware constraint workaround, or design pivot is made (e.g., memory layout, quantization format, zero-disk policy, fallback logic).
- **What to do**:
  - Append an ADR entry (`ADR-00X`) stating Title, Status, Context, Decision, and Consequences.

---

## 🛠️ Operating Principles

1. **Zero-Disk Model Storage**:
   Never download weights directly to local disk. Always use `tests/ephemeral_test_runner.py` or the Colab cloud tester (`notebooks/phantom_cloud_tester.ipynb`).
2. **Real over Synthetic**:
   Never introduce mocks or synthetic test data. All verification must run against real physical engines and true hardware telemetry.
3. **Clean Root Directory**:
   Do not dump scratch or specification markdown files into the workspace root. Place specs in `docs/specs/`, tests in `docs/testing/`, and scratch in `scratch/`.
