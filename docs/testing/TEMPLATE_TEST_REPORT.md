# Test [XX]: [Model Name] — Hardware Execution & Verification Report

**Audit Date**: YYYY-MM-DD  
**Target System**: [OS] | Host RAM: [XX] GB | GPU: [GPU Name] ([XX] GB VRAM)  
**Evaluator**: PHANTOM Automated Verification Suite / Antigravity Engineering  
**Test Objective**: [Brief description of the test objective, e.g. verify MoE sparse throughput or 70B layer streaming].

---

## 1. Executive Summary & Hardware Multipliers

* **Model Scale**: [XX.XX] Billion Parameters ([Dense / MoE Sparse with XX Active])
* **Quantization**: [Q4_K_M / Q8_0 / FP16]
* **Weight Footprint**: [XX.XX] GB resident across memory hierarchy
* **Scale Multiplier vs Native 4-bit VRAM Limit**: [XX.X×]
* **Scale Multiplier vs Native 16-bit VRAM Limit**: [XX.X×]

| Comparison Metric | Baseline Model / Limit | PHANTOM Live Executed Model | Authentic Multiplier |
|---|---|---|---|
| **vs Native 4-bit VRAM Limit** | ~7.0B – 8.0B Q4 (Fits in 6GB VRAM) | [Model Name] ([XX]B) | **[XX.X×] More Parameters** |
| **vs Native 16-bit VRAM Limit** | ~3.0B FP16 (Fits in 6GB VRAM) | [Model Name] ([XX]B) | **[XX.X×] More Parameters** |
| **Active Math vs Dense 32B** | 32.76B Dense (65.5 GFLOPs) | [Model Name] ([XX]B Active) | **[XX.X×] Active Compute Load** |

---

## 2. Real Hardware Load & Performance Metrics

| Metric | Measured Value | Target Baseline / Limit | Verdict |
|---|---|---|---|
| **GPU Dedicated VRAM Usage** | **[XXXX.X] MB** peak | [XXXX.X] MB physical limit | **[SAFE]** |
| **GPU Core Temperature** | **[XX.X°C – XX.X°C]** | 87.0°C thermal throttle threshold | **[OPTIMAL]** |
| **GPU Power Draw** | **[XX.X W]** | [XX.X W] TGP limit | **[EFFICIENT]** |
| **Host System RAM Usage** | **[XX.XX] GB** | [XX.XX] GB visible total | **[OPTIMAL]** |
| **Time-To-First-Token (Cold)** | **[XX.XX] seconds** | Initial load from storage | **[ONE-TIME]** |
| **Time-To-First-Token (Warm)** | **[X.XX] seconds** | Target < 3.0s | **[INTERACTIVE]** |
| **Decoding Throughput** | **[XX.XX] tokens/sec** | Target > [X.X] tok/s | **[STEADY]** |
| **System Stability** | **0 Crashes, 0 OOMs** | Windows commitment limit | **[100% STABLE]** |

---

## 3. Non-Synthetic Task Verification

### Task 1: Algorithmic Dynamic Programming (0/1 Knapsack)
* **Prompt**: *"Write a clean Python function `knapsack(weights, values, W)` using 1D space-optimized DP. What is the return value for weights=[10, 20, 30], values=[60, 100, 120], W=50? Give the final numeric answer clearly."*
* **Metrics**: [XXX] tokens | [X.XX] tok/sec | Peak VRAM: [XXXX] MB
* **Output Ground Truth Target**: `220`
* **Verdict**: **[PASS / FAIL]**

---

### Task 2: Mathematical Deduction (Harmonic Mean Velocity)
* **Prompt**: *"A train travels 120 miles from City A to City B at 60 mph, and immediately returns along the same 120-mile route from City B to City A at 40 mph. What is the average speed for the entire round trip? Show your calculation and give the final exact number."*
* **Metrics**: [XXX] tokens | [X.XX] tok/sec | Peak VRAM: [XXXX] MB
* **Output Ground Truth Target**: `48 mph`
* **Verdict**: **[PASS / FAIL]**

---

### Task 3: Code Synthesis & Whitespace Normalization
* **Prompt**: *"Write a concise Python function `reverse_words(s: str) -> str` that reverses the order of words in a string while compressing all consecutive spaces into a single space and removing leading/trailing spaces. Use standard python idiom."*
* **Metrics**: [XXX] tokens | [X.XX] tok/sec | Peak VRAM: [XXXX] MB
* **Output Ground Truth Target**: Idiomatic reverse word reconstruction
* **Verdict**: **[PASS / FAIL]**

---

## 4. Architectural Summary & Hardware Verdict

* **Primary Bottleneck**: [VRAM Bandwidth / Host RAM Bandwidth / NVMe Swap Latency]
* **Memory Allocation Map**:
  * GPU VRAM: [XX] layers ([X.XX] GB)
  * Host System RAM: [XX] layers ([XX.XX] GB)
  * NVMe Swap: [XX] layers ([XX.XX] GB)
* **System Status**: **[VERIFIED & RECORDED IN `docs/testing/INDEX.md`]**
