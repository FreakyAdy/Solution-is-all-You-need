# PHANTOM — Record of Corrections & Ground Truth Reconciliations (`CHANGES.md`)

This document is a public, transparent record of all historical claims, metrics, and architecture assumptions in PHANTOM that have been audited, corrected, restated, or eliminated.

---

## Why This Document Exists

In early iterations of PHANTOM, several performance claims were based on analytic formulas, synthetic random-tensor benchmarks, or theoretical targets rather than live end-to-end measurements on physical silicon. Furthermore, single metrics drifted across different documents (for example, Wraith latency appearing variously as 0.487 ms, 0.458 ms, and 0.71 ms).

As part of the **Ground Truth Remediation** (September 15, 2026), we conducted a full physical audit on named hardware (**NVIDIA GeForce RTX 4050 Laptop GPU, 24 GB Host RAM, Gen4 NVMe SSD**). We deleted all self-generated grading audits, purged synthetic mocks, built automated byte accounting, and reconciled every claim against physical hardware limits.

---

## 1. Summary of Claim Corrections

| Original Claim | Historical Published Value | Corrected Ground Truth Value | Action Taken | Technical Rationale |
|---|---|---|:---:|---|
| **Llama-3-70B Throughput on RTX 4050 Laptop** | ~3.5 tok/sec | **0.39 tok/sec** (NVMe 3-tier swap) | **CORRECTED** | A 70B Q4_K_M model requires ~37 GB. Laptop fast tier (6GB VRAM + 22GB RAM) is 28 GB; ~12–15 GB must stream from NVMe on every token. At 1.4–1.8 GB/s NVMe read speed, throughput is physically capped at ~0.12–0.39 tok/s. The 3.5 tok/s claim violated physical storage bandwidth. |
| **Qwen2.5-Coder-32B PCIe Weight Streaming Paradox** | ~0.8 tok/s PCIe ceiling vs 2.88 tok/s | **2.88 tok/s via in-place DDR5 RAM execution** | **EXPLAINED & VERIFIED** | PHANTOM does *not* stream 15 GB of weights over PCIe every token. Layers 14–63 execute in-place in Host RAM on the CPU using SIMD kernels at DDR5 memory bandwidth (~48 GB/s). Only intermediate activation tensors (~10 KB) cross PCIe ($1.3\ \mu\text{s}$ transfer). |
| **Hardware Capacity Multiplier ("101.3B Limit")** | +10.6× ceiling lift (101.3B) | **Replaced with physical memory budgeting in `phantom plan`** | **DELETED FORMULA** | `101.3B` was calculated from an unvalidated algebraic formula in `hardware_detect.py` that credited free NVMe space as model capacity without accounting for the resulting 0.05 tok/s speed penalty. |
| **Chronos Scheduler Context Switch Latency** | 80.5 ms / 80.9 ms full context switch | **80.5 ms for in-RAM pointer/KV swap; cold swap requires 5.2–11.4s** | **RESTATED** | 80 ms only applies when both models already reside in Host RAM (swapping active pointers and KV slots). A cold switch requiring weight streaming from NVMe takes seconds and is now explicitly documented. |
| **Wraith Prefetch Predictor Latency & Hit Rate** | 0.458 ms / 0.71 ms / 100% accuracy | **0.438 ms latency; 92.4% accuracy; 9.9% end-to-end throughput lift** | **REBENCHMARKED** | The old benchmark measured a synthetic loop. The new benchmark measures real predictor execution time distribution and reports an ablation showing prefetch-enabled vs disabled decoding speeds. |
| **Neural Cache KV Compression & Error** | 8.0× compression, 1.07% / 1.15% error | **8.0× compression, 1.05 ± 0.05% error, verified on 64K context** | **REBENCHMARKED** | Replaced wobbling single numbers with empirical distributions ($N \ge 10$) and declared what the benchmark proves vs does not prove. |
| **Phantom Pages 64MB Layer Tile Read Latency** | 43.6 ms / 47.2 ms | **38.4 ± 3.1 ms (1.4–1.8 GB/s sustained)** | **REBENCHMARKED** | Measured sustained read throughput over repeated iterations on local Gen4 NVMe storage. |
| **RTX 4090 / 200B Parameter Claim** | Running 200B on desktop RTX 4090 | **Removed from verified claims** | **DELETED** | We do not own an RTX 4090 in this test environment. All claims about hardware not physically tested have been deleted or moved to speculative projections. |
| **Competitor Comparisons (Ollama / vLLM / HF)** | Ollama 0.0 tok/s OOM, vLLM OOM | **Labeled as architectural requirements rather than benchmarks** | **RESTATED** | vLLM requires $\ge 24$ GB VRAM by design and does not boot on 6GB VRAM. This is documented as an architectural memory minimum rather than an executed benchmark. |

---

## 2. Structural Guardrails Implemented

To ensure these drifting numbers can never recur:

1. **Deletion of `audit.md`**: The 369-line self-grading audit document has been permanently deleted.
2. **Programmatic Results (`RESULTS.md`)**: Generated exclusively by `scripts/generate_results.py` from `benchmarks/results/latest.json`. Manual edits are forbidden and enforced via `--check` in CI.
3. **No Single Numbers in Prose**: All benchmarked metrics are reported as distributions (mean, stddev, min, max, p50, p95) across $N \ge 10$ iterations with environment fingerprints.
4. **Mandatory Proves / Does Not Prove Declarations**: Every innovation explicitly states its physical boundary and what it cannot overcome.
5. **Claims Checker CI Gate**: `scripts/check_claims.py` scans all markdown files to ensure every numeric claim matches `latest.json` or is explicitly justified in `docs/claims_allowlist.yml`.
