You are a senior systems auditor performing a complete technical audit of PHANTOM CORE — an open-source LLM inference engine that claims to run models far beyond a GPU's native VRAM capacity through 7 original innovations. Your job is NOT to be impressed. Your job is to verify that everything compiles, integrates correctly, hits its stated benchmarks, and is architecturally sound. You will find every bug, every broken assumption, every unverified claim, every missing integration, and every performance lie. You will also identify missing features that would make this genuinely valuable as an open-source project.

Work through ALL sections below. Do not skip any section. For each item, output: [PASS], [FAIL: reason], [WARN: reason], or [MISSING] depending on what you find. At the end, produce a prioritized fix list sorted by severity (CRITICAL → HIGH → MEDIUM → LOW).

---

## SECTION 1 — COMPILATION & BUILD SYSTEM AUDIT

### 1.1 CUDA Kernels (kernels/)
- [ ] Does dct_compress.cu compile cleanly with CUDA 12.x? Run: nvcc -arch=sm_86 --std=c++17 kernels/spectral_quant/dct_compress.cu -o /tmp/dct_test. Report all warnings and errors.
- [ ] Does idct_reconstruct.cu exist as a separate file OR is the iDCT fused into dct_compress.cu as promised? Verify the design decision is consistent with the CMakeLists.txt.
- [ ] Does fisher_calibrate.cu implement Fisher Information correctly? Verify: Fisher diagonal = E[(∂L/∂w)²]. Check that the gradient computation is done without autograd — this must be a pure CUDA kernel computing per-weight squared gradient norms.
- [ ] Do kv_encode.cu and kv_decode.cu compile? Does kv_decode.cu actually fuse the decompression + attention score computation + softmax in a single kernel, or does it decompress to a temp buffer first (which would defeat the purpose)?
- [ ] Does gate_predict.cu implement a linear probe (NOT an MLP) as specified? Verify the gate architecture matches: gate = sigmoid(W_gate @ x + b_gate) exactly.
- [ ] Does sparse_matmul.cu correctly fall back to dense cuBLAS GEMM when sparsity < 30%? Verify the threshold check exists and the fallback doesn't introduce a measurable overhead.
- [ ] Does flash_attn_v3.cu implement FlashAttention-2 style O(1) HBM IO complexity, or is it an approximation? Verify tiling strategy.
- [ ] Does gqa_kernel.cu correctly handle cases where num_kv_heads < num_query_heads? Verify the key/value head broadcasting logic.
- [ ] CMakeLists.txt: Does it auto-detect the GPU's compute capability (sm_XX) and compile for it? Verify it does NOT hardcode sm_86 or any specific arch.
- [ ] Do ALL CUDA kernels have `CUDA_CHECK()` error checking on every CUDA API call? Grep for unchecked cudaMalloc, cudaMemcpy, kernel launches (<<<>>>).

### 1.2 Rust Core (core/)
- [ ] Does `cargo build --release` complete with zero errors?
- [ ] Does Cargo.toml include all required dependencies: tokio (async runtime), tokio-uring (io_uring for NVMe), lz4 (compression), pyo3 (Python FFI for Wraith LSTM), serde + rmp-serde (MessagePack), tracing (structured logging)?
- [ ] Does phantom_pages.rs use tokio-uring correctly for async io_uring ops, or is it using std::fs (blocking I/O) disguised as async? This is a critical performance difference.
- [ ] Does the LRU map in lru_map.rs persist to ~/.phantom/lru_state.msgpack between restarts? Verify by running the engine twice and checking if layer residency preferences from session 1 influence session 2.
- [ ] Does engine.rs compile with the PyO3 bindings to the Wraith LSTM? Run `cargo test` and verify no linker errors from the pyo3 FFI boundary.
- [ ] Does resonance.rs actually read GPU thermal state (nvmlDeviceGetTemperature) or does it use a placeholder? Check for NVML calls.
- [ ] Does chronos.rs implement a real sub-400ms model context switch or is it a stub? Verify the eviction + staging + reload pipeline exists.
- [ ] Does the IPC server in server.rs correctly handle concurrent requests with a queue (not blocking the entire engine on a single generation)?
- [ ] Are ALL Rust public functions and structs documented with doc-comments (/// syntax)?
- [ ] Does `cargo clippy -- -D warnings` pass clean?

### 1.3 Python Layer (python/)
- [ ] Does `pip install -e .` complete without errors? Check pyproject.toml for all required dependencies: torch>=2.3, fastapi, uvicorn, structlog, numpy, scipy (for DCT reference).
- [ ] Does wraith_lstm.py implement the LSTM on CPU as specified (not GPU)? Verify by checking device placement in __init__.
- [ ] Does wraith_lstm.py's predict_next() complete in < 1ms on CPU? Benchmark: run 1000 iterations and report p50/p95/p99 latencies.
- [ ] Does neural_cache_ae.py train the 3-layer autoencoder to < 2% reconstruction error in the 4-minute time budget? Run against a small test model and report final validation loss.
- [ ] Does run_calibration.py complete all 5 steps (A through E) in under 10 minutes on a 7B model? Time each step individually.
- [ ] Does auto_detect.py correctly identify the GPU model, VRAM, PCIe gen/width, RAM amount, and NVMe speed? Run and verify against `nvidia-smi`, `lspci`, and `lsblk` output.
- [ ] Does openai_compat.py implement ALL four endpoints: POST /v1/chat/completions (streaming + non-streaming), POST /v1/completions, GET /v1/models, GET /v1/health, GET /v1/metrics?
- [ ] Does the /v1/metrics endpoint return ALL specified fields (vram_mb, ram_mb, nvme_mb, layer_residency, wraith_accuracy_pct, kv_compression_ratio, active_sparsity_pct, tok_per_sec, thermal_state, throttle_active)?

### 1.4 UI (ui/)
- [ ] Does `npm install && npm run build` complete without errors?
- [ ] Does LayerMap.tsx render the layer grid as a 2D sqrt(N) x sqrt(N) layout (NOT a vertical list)?
- [ ] Does the layer map update every 200ms via WebSocket? Verify the WebSocket connection is established to the metrics endpoint.
- [ ] Does clicking a cell in LayerMap correctly send a force-pin command to the Rust core and does the pin persist?
- [ ] Does VRAMGauge.tsx show all three tiers (VRAM, RAM, NVMe) simultaneously?
- [ ] Does the CalibrationWizard correctly run the Python calibration pipeline and show progress? Verify it doesn't just show a spinner — it should stream calibration step completion events.

### 1.5 Installer (install.sh)
- [ ] Does install.sh detect if NVIDIA GPU is missing and exit with a helpful error?
- [ ] Does it auto-detect compute capability and pass the correct -arch=sm_XX to CMake?
- [ ] Does it print the correct native ceiling and Phantom ceiling for the detected hardware?
- [ ] Does it handle the case where NVMe space is < 50GB (warn and adjust phantom_swap.bin size)?
- [ ] Does it correctly register the systemd service without requiring root for the service unit itself?
- [ ] Does it work on Ubuntu 22.04 AND WSL2 (check for WSL2 detection and skip systemd registration)?

---

## SECTION 2 — INNOVATION CORRECTNESS AUDIT

For each claimed innovation, verify the implementation is actually what was described — not a simplified approximation.

### 2.1 WRAITH LAYERS — Predictive Layer Prefetching
- [ ] ARCHITECTURE CHECK: Is the LSTM exactly 2-layer, hidden_dim=64, bidirectional=False as specified? Print model.parameters() count — it should be approximately 200 parameters.

  VERIFY THIS MATH: A 2-layer LSTM with input_size=3, hidden_size=64:
  Layer 1: 4 * (64*(3+64) + 64) = 4 * (4288 + 64) = 4 * 4352 = 17408 params
  Layer 2: 4 * (64*(64+64) + 64) = 4 * (8192 + 64) = 4 * 8256 = 33024 params
  Output linear: 64 * num_layers + num_layers biases
  Total is ~50K+ params for a 80-layer model — NOT 200 parameters as stated.
  The 200-parameter claim in the build prompt is mathematically incorrect for an LSTM on multi-layer inputs.
  REPORT: Does the actual implementation match the architecture, or was the param count claim wrong?

- [ ] ONLINE LEARNING: Does update_online() actually perform backprop through the last 200 steps, or does it just update from the current step? Verify by checking if there's a replay buffer of observation history.
- [ ] CONVERGENCE: Before the LSTM converges (first 50 tokens of a new model), does predict_next() return a reasonable fallback (e.g., sequential layer access)? Verify the cold-start behavior.
- [ ] INTEGRATION: Is the Wraith predictor's output actually consumed by the PhantomPageManager's prefetch_hint()? Trace the call chain from LSTM output → Rust prefetch queue.
- [ ] ACCURACY MEASUREMENT: How is wraith_accuracy_pct computed? Verify it measures P(predicted_layers ∩ actually_accessed_layers) / P(actually_accessed_layers) — i.e., recall of prefetch hits, not just any intersection.

### 2.2 SPECTRAL QUANTIZATION — DCT Weight Compression
- [ ] CORRECTNESS: Apply the forward DCT and inverse DCT to a known test matrix. Verify reconstruction error is < 0.01% when K = N (all coefficients retained).
- [ ] K-SELECTION: Does the Fisher Information approximation for K selection actually compute per-layer importance weights, or is it a uniform K across all layers? Verify per-layer K values differ meaningfully.
- [ ] PERPLEXITY CLAIM: Run perplexity on a standard dataset (WikiText-2) comparing FP16 baseline vs Spectral Quant. Report actual PPL delta — does it stay within ≤ 1.2 PPL?
- [ ] FP8 ENCODING: What FP8 format is used (E4M3 or E5M2)? Verify the range is appropriate for DCT coefficients (which can span a much wider range than activations — DCT coefficients are NOT bounded like weights in [−1,1]).
- [ ] PERFORMANCE: Time the DCT compression kernel on a 4096x14336 matrix on the target GPU. Does it complete in < 8ms as claimed?
- [ ] RECONSTRUCTION AT INFERENCE: Verify that weight reconstruction (iDCT) happens at inference time per-forward-pass, not once at model load. This is the correct design (reconstruct → compute → discard). If reconstruction is cached, you've lost the memory savings.

### 2.3 NEURAL CACHE — Learned KV Compression
- [ ] AUTOENCODER ARCHITECTURE: Verify the encoder is Linear(D, D//4) → GELU → Linear(D//4, D//8) and decoder is the reverse. For LLaMA-70B where D=8192, this means compressing 8192→1024. Verify dimensions are correct per-model.
- [ ] TRAINING DATA: Does the training use actual KV activations collected from the model (not random tensors)? Verify calibration step C collects real activations.
- [ ] RECONSTRUCTION ERROR: Measure actual cosine distance between original KV entries and reconstructed ones. Does it stay ≤ 2%?
- [ ] FUSED KERNEL: This is the hardest claim. Does kv_decode.cu actually fuse decompress + attention score + softmax without materializing full KV? Or does it decompress to a temp FP16 buffer first? The latter is a correctness shortcut that wastes memory.
- [ ] GQA HANDLING: For models with GQA (e.g., LLaMA-3-70B uses 8 KV heads for 64 query heads), does the fused kernel correctly broadcast compressed KV heads across query groups?
- [ ] CONTEXT LENGTH EFFECT: Verify the claimed 8× context extension. If you normally fit 4096 tokens of KV cache, can you now fit 32768 tokens? Measure actual max context before OOM.

### 2.4 PHANTOM PAGES — NVMe Virtual VRAM
- [ ] IO_URING USAGE: Verify the Linux implementation uses io_uring (via tokio-uring), not epoll or blocking reads. Check: strace -e trace=io_uring_setup,io_uring_enter phantom-core during NVMe load and confirm uring syscalls appear.
- [ ] PRE-ALLOCATION: Does phantom_swap.bin get pre-allocated at install time (fallocate), or does it grow dynamically? Dynamic growth causes fragmentation and defeats the sequential-read optimization.
- [ ] COMPRESSION: Verify LZ4 compression is applied to BF16 tensors before NVMe write. Measure actual compression ratio on transformer weight layers (expect ~1.5-2.5× for float tensors).
- [ ] < 50ms CLAIM: Load a single layer (approx 500MB for a 70B model layer) from NVMe Gen4. Does it complete in < 50ms? Note: at 7GB/s sequential NVMe speed, 500MB should take ~71ms. The claim may be physically impossible for full layers. Report actual measured time.
- [ ] LRU PERSISTENCE: Write 10 layers, access them in a specific order, shutdown the engine, restart, verify the hot/cold map matches the previous session's access pattern.
- [ ] 512MB PHANTOM BLOCKS: Verify layers are packed into 512MB blocks as recommended (for sequential read efficiency), not stored one layer per file.

### 2.5 ADAPTIVE COMPUTE ROUTING — Sparse Activation
- [ ] GATE ARCHITECTURE: The gate is a 16-parameter linear probe per MLP block. For a 70B model with D=8192 and D_ffn=28672, W_gate would be [D_ffn × D] = [28672 × 8192] which is 235M parameters — clearly NOT 16 parameters. REPORT: What is the actual gate architecture? The 16-parameter claim is impossible for the described function.
- [ ] SPARSITY MEASUREMENT: Measure actual neuron activation sparsity on your target model+dataset. Does it fall in the claimed 30-70% range? Report per-layer sparsity histogram.
- [ ] PERFORMANCE CROSSOVER: Verify sparse_matmul.cu outperforms dense cuBLAS GEMM at > 40% sparsity. Run both at 40%, 50%, 60%, 70% sparsity and report GFLOP/s.
- [ ] GATE CALIBRATION: Report F1 score of gate predictions on held-out activations. Does it exceed 0.85 precision as claimed? Report per-layer F1 distribution.
- [ ] FALLBACK: For layers where F1 < 0.70, verify the gate is disabled and dense compute is used (no correctness regression from wrong sparsity predictions).

### 2.6 CHRONOS SCHEDULER — Multi-Model Execution
- [ ] CONTEXT SWITCH TIME: Measure actual model context switch time between two loaded models. Does it complete in < 400ms? Run 100 switches and report p50/p95/p99.
- [ ] STATE CHECKPOINTING: When switching from Model A to Model B, does Chronos save Model A's KV cache state? If not, resuming Model A will require re-prefill which is unacceptable.
- [ ] MEMORY ACCOUNTING: Verify that with two models loaded, total memory usage (VRAM + RAM + NVMe) stays within hardware limits. The scheduler should never exceed available tiers.
- [ ] CONCURRENT REQUEST HANDLING: Send simultaneous requests to Model A and Model B via the API. Verify they are queued correctly and the second request waits for the context switch, not for the first to finish completely.

### 2.7 RESONANCE SAMPLER — Hardware-Aware Sampling
- [ ] NVML INTEGRATION: Verify actual NVIDIA Management Library calls for temperature, power, and memory bandwidth utilization. Test with GPU at < 70°C and > 85°C — does sampling behavior change?
- [ ] THERMAL THROTTLE DETECTION: Verify the sampler detects nvmlClocksThrottleReasonHwThermal correctly and reduces beam width in response.
- [ ] QUALITY CONSISTENCY: Run the same prompt 20 times — 10 under low load (GPU idle) and 10 under high load (GPU thermal-throttled). Does output quality remain perceptually consistent? This is the core claim.
- [ ] PCIe SATURATION DETECTION: How does the sampler detect PCIe saturation? NVML does not expose PCIe utilization directly. Verify the implementation uses a proxy metric (e.g., memory bandwidth utilization, PCIe counter via pynvml) or flag this as unimplemented.

---

## SECTION 3 — INTEGRATION AUDIT

These tests verify that the components actually talk to each other correctly.

### 3.1 Python ↔ Rust IPC
- [ ] Send a /v1/chat/completions request and verify the Python API server correctly serializes it to MessagePack and sends via Unix socket to the Rust engine.
- [ ] Verify the Rust engine's response (streaming tokens) is correctly received and forwarded as SSE (Server-Sent Events) to the HTTP client.
- [ ] Test with a malformed request — verify the Rust engine returns a PhantomError that propagates as a 400/500 HTTP response, not a panic.
- [ ] Test IPC under load: 5 concurrent requests. Verify request queuing works and responses are correctly correlated with requestors (no response mixing).

### 3.2 Wraith LSTM ↔ PhantomPageManager
- [ ] Verify: after predict_next() returns layer_ids [42, 43, 44], these IDs are passed to prefetch_hint() before layer 42 is needed by the forward pass.
- [ ] Simulate a Wraith miss (predictor wrong): verify the engine falls back to synchronous load without crashing. Measure added latency for a miss vs a hit.
- [ ] Verify Wraith predictions improve over a 100-token generation. Report prefetch hit rate at tokens 1-10, 11-50, 51-100.

### 3.3 Calibration Profile ↔ Inference Engine
- [ ] Load a .phantom profile and verify the engine correctly applies: (a) per-layer K values from spectral_k_map.json, (b) gate weights from gate_weights.npz, (c) KV autoencoder from kv_ae_cuda_weights.bin.
- [ ] Run the same prompt with and without a calibration profile. Verify: with profile → lower VRAM usage, with profile → higher perplexity (small but measurable regression expected).
- [ ] Verify model hash in the profile matches the loaded model — the engine should reject a profile for a different model.

### 3.4 UI ↔ Metrics Endpoint
- [ ] Open the Electron UI and load a model. Verify LayerMap cells update every 200ms and correctly reflect VRAM/RAM/NVMe residency.
- [ ] Pin a layer via UI click. Verify the pin is sent to the engine AND the layer is not evicted during a generation that would normally evict it.
- [ ] Simulate GPU thermal throttle (stress test the GPU). Verify ThermalMonitor.tsx shows the correct throttle state within one polling cycle.

---

## SECTION 4 — PERFORMANCE BENCHMARK EXECUTION

Run each benchmark and report actual numbers vs targets:

### 4.1 bench_spectral_quant.py
Target: ≤ 1.2 PPL delta vs FP16 on WikiText-2.
Run: python tests/benchmarks/bench_spectral_quant.py --model <test_model>
Report: [baseline_ppl] vs [spectral_ppl], delta = [X], [PASS/FAIL]

### 4.2 bench_wraith_prefetch.py
Target: ≥ 80% prefetch hit rate after warm-up.
Run: python tests/benchmarks/bench_wraith_prefetch.py
Report: [hit_rate_at_token_10] [hit_rate_at_token_50] [hit_rate_at_token_100], [PASS/FAIL]

### 4.3 bench_phantom_pages.py
Target: ≤ 50ms per layer load from NVMe Gen4.
Run: python tests/benchmarks/bench_phantom_pages.py --tier auto
Report: [p50_ms] [p95_ms] [p99_ms] per-layer load time, [PASS/FAIL]
IMPORTANT: Also report the physical limit: (layer_size_MB / nvme_seq_bandwidth_GBps) × 1000 = theoretical minimum ms. If measured time < theoretical minimum, there is a bug in the measurement (data was cached in OS page cache, not actually read from NVMe).

### 4.4 bench_chronos.py
Target: ≤ 400ms model context switch.
Run: python tests/benchmarks/bench_chronos.py
Report: [p50_ms] [p95_ms] [p99_ms] context switch time, [PASS/FAIL]

### 4.5 bench_full_pipeline.py --tier auto
This is the main claim benchmark.
For the detected hardware tier, report:
- What model size was used as the "above native ceiling" test case
- Actual tok/sec achieved
- Actual VRAM used during generation
- Whether the run completed without OOM
- Compare: native llama.cpp on same hardware, same model (if feasible) — report PHANTOM CORE overhead vs baseline

---

## SECTION 5 — CORRECTNESS REGRESSION TESTS

### 5.1 Output Determinism
- [ ] Generate 3 completions for the same prompt with the same seed, temperature=0. Verify all 3 outputs are byte-identical. Non-determinism in the sparse matmul or fused attention kernels would cause failures here.

### 5.2 Optimization Correctness vs Baseline
- [ ] Generate a completion with ALL optimizations disabled (vanilla FP16 inference). Generate the same with ALL optimizations enabled. Compute cosine similarity of logit distributions at each token position. Report mean cosine similarity — should be > 0.995 for the optimizations to be acceptable.

### 5.3 Long Context Test
- [ ] Load a 32K token document into the KV cache with Neural Cache enabled. Verify the model correctly answers questions about content from the beginning of the document (i.e., the KV autoencoder hasn't lost information from early tokens). Use a needle-in-haystack test.

### 5.4 Sparsity Correctness
- [ ] Run a known mathematical computation (e.g., a prompt that requires exact arithmetic) with sparse routing enabled vs disabled. Verify outputs match (or differ within acceptable tolerance for the mathematical task).

### 5.5 Multi-Model Isolation
- [ ] Load Model A and Model B simultaneously. Generate from Model A, then switch to Model B, then switch back to Model A. Verify Model A's output after switching back is identical to what it would have been without the switch (i.e., KV cache state was correctly preserved and restored).

---

## SECTION 6 — OPEN SOURCE READINESS AUDIT

### 6.1 Missing Critical Features for an Open Source Project
The build prompt describes a complete system but omits several things that would make this actually usable by the community:

- [ ] MISSING: Model download integration. The ModelManager UI mentions downloading models but the build prompt has no downloader. HuggingFace Hub Python client integration is needed.
- [ ] MISSING: GGUF format support. The loader.py mentions GGUF but the spectral quantization pipeline assumes raw weight tensors. GGUF files store pre-quantized weights (GPTQ, Q4_K_M, etc.) — applying DCT on top of already-quantized weights is mathematically questionable. Clarify: does PHANTOM CORE work with base (FP16/BF16) models only?
- [ ] MISSING: Windows native support. The build prompt mentions WSL2 but io_uring is Linux-only. On native Windows, the NVMe tier would need a different I/O strategy (IOCP or simple async reads). The current design makes Windows support a first-class lie unless WSL2 is the only supported path.
- [ ] MISSING: Model architecture auto-detection edge cases. auto_detect.py needs to handle Mixtral (MoE architecture — the MLP sparsity assumptions break for mixture-of-experts since experts are ALREADY sparse), Phi-3 (different attention pattern), Command-R (long context base), and Gemma-2.
- [ ] MISSING: Calibration profile sharing. Users should be able to download pre-built .phantom profiles for popular models instead of running 10-minute calibration themselves. There is no profile registry or CDN in the design.
- [ ] MISSING: Quantization pipeline for the user. The spec assumes models arrive as FP16/BF16 base weights. In practice, most users have GGUF or GPTQ models. Either a dequantize → re-quantize (spectral) pipeline is needed, or explicit GGUF passthrough mode.
- [ ] MISSING: Multi-GPU support. The architecture is entirely single-GPU. The design doesn't mention tensor parallelism or pipeline parallelism for systems with 2-4 consumer GPUs.
- [ ] MISSING: CPU-only fallback. For users with no NVIDIA GPU (AMD, Apple Silicon, CPU-only), PHANTOM CORE simply fails. ROCm support or a CPU fallback path should be documented even if not implemented.
- [ ] MISSING: Contribution guidelines, code of conduct, issue templates, PR templates (standard OSS hygiene but important for adoption).
- [ ] MISSING: Safety and content filtering hooks in the API layer. An open-source inference engine deployed as a local API server should document its stance on safety filtering (even if the answer is "none by default, here's how to add it").

### 6.2 Documentation Completeness
- [ ] ARCHITECTURE.md: Does it explain the data flow of a single forward pass through all 7 innovations? Draw an ASCII or Mermaid diagram.
- [ ] INNOVATIONS.md: Does it compare each innovation to the most similar existing work (e.g., Spectral Quant vs SqueezeLLM, Neural Cache vs H2O/SnapKV, Phantom Pages vs llama.cpp mmap)?
- [ ] INSTALL.md: Does it list ALL prerequisites with exact version numbers?
- [ ] API.md: Does it document the /v1/metrics extension fields and the WebSocket streaming format?

### 6.3 License
- [ ] Is a LICENSE file present?
- [ ] Are all third-party dependencies (CUDA kernels, PyTorch, etc.) license-compatible with the chosen license?

---

## SECTION 7 — SECURITY & STABILITY AUDIT

- [ ] Does the IPC server validate input size before deserializing MessagePack? A malformed large packet could cause a memory allocation panic.
- [ ] Does the NVMe page manager handle the case where phantom_swap.bin is corrupted (e.g., interrupted write)? Verify it detects corruption via a stored checksum per page.
- [ ] Does the calibration pipeline handle models that produce NaN activations (common in poorly quantized models)? Verify it fails gracefully with a diagnostic instead of training a corrupt autoencoder.
- [ ] Does the Rust engine panic on CUDA kernel errors, or does it recover and report the error through the Result chain? Grep for unwrap() calls in engine.rs and report count.
- [ ] Does the API server have request size limits to prevent OOM from a malicious very long prompt?

---

## SECTION 8 — SUGGESTED ADDITIONS FOR V1.0 OSS RELEASE

After completing the audit above, evaluate and implement the following enhancements if missing:

### High-Value Additions
1. **Profile Registry Protocol** — A simple ~/.phantom/registry.json that maps model hashes to profile download URLs. Community members can share calibration profiles. First request for an uncalibrated model checks the registry before running local calibration.

2. **FlashInfer Integration** — Instead of maintaining a custom flash_attn_v3.cu, integrate FlashInfer (the production-grade attention kernel library) for attention computation. Keep only the Neural Cache fused kernel as the custom CUDA component. This reduces maintenance burden and improves compatibility with future GPU architectures.

3. **Prometheus Metrics Exporter** — Add a /metrics endpoint in Prometheus format (in addition to the JSON /v1/metrics). This lets users integrate PHANTOM CORE into Grafana dashboards out of the box.

4. **GGUF Dequantize → Spectral Requantize Pipeline** — A one-time conversion tool: phantom convert --input model.gguf --output model.phantom that dequantizes GGUF weights to FP32, applies Spectral Quantization, and produces a PHANTOM CORE native format. This is critical for community adoption since most available models are GGUF.

5. **Benchmark Leaderboard in README** — A table showing measured tok/sec on different hardware tiers (contributed by the community). Include the hardware profiling script that users can run and submit results.

6. **Graceful Degradation Mode** — A --safe-mode flag that disables Spectral Quant and Neural Cache (the lossy innovations) and only uses Phantom Pages + Wraith Layers (lossless memory expansion). This gives users a way to verify correctness before enabling lossy compression.

7. **ROCm Stub** — Even if not implemented, add a kernels/rocm/ directory with stub files and a clear README explaining the porting path. AMD GPU users will attempt to use this; guide them rather than letting them hit a wall.

8. **Quantization-Aware Calibration** — The current calibration collects activations from FP16 inference. For users who want to stack Spectral Quant on top of INT8 quantization, the calibration needs to run on the INT8 model to correctly characterize the actual activation distributions the kernels will see.

9. **phantom doctor Command** — A diagnostic CLI command that runs a suite of self-tests:
   - Checks CUDA kernel correctness (reference vs kernel output)
   - Verifies NVMe I/O speed meets minimum requirements
   - Validates a loaded .phantom profile against the loaded model
   - Reports any mismatches with actionable fix suggestions

10. **Memory Budget Planner** — A CLI tool (phantom plan --model llama3-70b --vram 6 --ram 32 --nvme 200) that, WITHOUT loading the model, computes: expected tok/sec, expected VRAM/RAM/NVMe distribution, expected calibration time, and whether the target model fits with comfortable headroom. This is the first thing a new user would want to run.

---

## FINAL OUTPUT FORMAT

After completing all sections above, produce:

### Summary Report

PHANTOM CORE AUDIT REPORT
Generated: [timestamp]
Hardware: [GPU model, VRAM, RAM, NVMe]
Phantom Core Version: [git hash]

SECTION RESULTS:
Section 1 (Build): [X/Y PASS, Z FAIL, W WARN]
Section 2 (Innovations): [X/Y PASS, Z FAIL, W WARN]
Section 3 (Integration): [X/Y PASS, Z FAIL, W WARN]
Section 4 (Benchmarks): [X/Y PASS, Z FAIL, W WARN]
Section 5 (Correctness): [X/Y PASS, Z FAIL, W WARN]
Section 6 (OSS Readiness):[X/Y PASS, Z FAIL, W WARN]
Section 7 (Security): [X/Y PASS, Z FAIL, W WARN]

OVERALL: [READY TO SHIP / NEEDS WORK / BLOCKED]


### Prioritized Fix List

CRITICAL (blocks basic function):
[1] [component] — [problem] — [fix]
...

HIGH (major claim is wrong or broken):
[1] [component] — [problem] — [fix]
...

MEDIUM (works but not as described):
[1] [component] — [problem] — [fix]
...

LOW (polish, docs, nice-to-have):
[1] [component] — [problem] — [fix]
...


### Architecture Inconsistencies Found
List any places where two parts of the spec contradict each other (e.g., the 16-parameter gate claim vs the mathematical impossibility of that for large D_ffn values).

### Performance Claims Verified
For each benchmark, state: [VERIFIED / NOT MET / PHYSICALLY IMPOSSIBLE / NOT TESTED].
The "physically impossible" category is important — some claims in the build prompt may exceed hardware physical limits (e.g., the 50ms NVMe layer load for a 500MB layer on Gen4 SSD).