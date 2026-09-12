/**
 * @file gate_predict.cu
 * @brief PHANTOM CORE — Sparsity Gate Prediction Kernel
 *
 * Innovation 5: Adaptive Compute Routing — Sparse Activation Exploitation
 *
 * Predicts which MLP neurons will activate for a given input using a trained
 * linear gate: gate = sigmoid(W_gate @ x + b_gate)
 * 
 * Outputs a binary mask indicating active neurons (gate > threshold).
 * When sparsity > 40%, the subsequent sparse_matmul kernel skips inactive
 * neurons entirely, yielding dramatic FLOP reduction.
 *
 * Copyright (c) 2025 PHANTOM CORE Project
 */

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

constexpr int GATE_BLOCK_SIZE = 256;

// ============================================================================
// KERNEL: GATE PREDICTION WITH SIGMOID + THRESHOLDING
// ============================================================================

/**
 * @brief Predict neuron activation mask using trained linear gate.
 *
 * @param input          Input activation vector [B, D] in FP16
 * @param gate_weights   Gate weight matrix [D_ffn, D] in FP16
 * @param gate_bias      Gate bias vector [D_ffn] in FP16
 * @param gate_output    Raw gate values (pre-threshold) [B, D_ffn] in FP32
 * @param active_mask    Binary activation mask [B, D_ffn] in uint8
 * @param active_count   Number of active neurons per sample [B]
 * @param active_indices Indices of active neurons [B, D_ffn] (packed, variable length)
 * @param B              Batch size
 * @param D              Input dimension
 * @param D_ffn          FFN/MLP intermediate dimension
 * @param threshold      Activation threshold (default 0.5)
 *
 * Grid: (B, cdiv(D_ffn, blockDim.x))
 * Block: (GATE_BLOCK_SIZE)
 */
__global__ void gate_predict_kernel(
    const half* __restrict__ input,
    const half* __restrict__ gate_weights,
    const half* __restrict__ gate_bias,
    float* __restrict__ gate_output,
    uint8_t* __restrict__ active_mask,
    int* __restrict__ active_count,
    int B,
    int D,
    int D_ffn,
    float threshold
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int neuron = blockIdx.y * blockDim.x + threadIdx.x;
    if (neuron >= D_ffn) return;

    const half* x = input + (size_t)b * D;
    const half* w_row = gate_weights + (size_t)neuron * D;

    // Compute gate value: sigmoid(w @ x + b)
    float acc = __half2float(gate_bias[neuron]);
    for (int d = 0; d < D; d++) {
        acc += __half2float(x[d]) * __half2float(w_row[d]);
    }

    // Sigmoid
    float gate_val = 1.0f / (1.0f + expf(-acc));
    gate_output[(size_t)b * D_ffn + neuron] = gate_val;

    // Threshold
    uint8_t is_active = (gate_val >= threshold) ? 1 : 0;
    active_mask[(size_t)b * D_ffn + neuron] = is_active;

    // Count active neurons using atomic add
    if (is_active) {
        atomicAdd(&active_count[b], 1);
    }
}

// ============================================================================
// KERNEL: COMPACT ACTIVE INDICES
// ============================================================================

/**
 * @brief Build a compact list of active neuron indices from the mask.
 *
 * After gate_predict_kernel sets the mask, this kernel compacts the
 * active indices into a dense array for efficient sparse matmul dispatch.
 *
 * @param active_mask     Binary mask [B, D_ffn]
 * @param active_indices  Output: packed active indices [B, D_ffn]
 * @param active_count    Output: count per sample [B] (reset and recomputed)
 * @param B               Batch size
 * @param D_ffn           FFN dimension
 */
__global__ void compact_active_indices_kernel(
    const uint8_t* __restrict__ active_mask,
    int* __restrict__ active_indices,
    int* __restrict__ active_count,
    int B,
    int D_ffn
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Single thread per batch element builds the compact index list
    // (parallelizable with prefix sum, but D_ffn is small enough ~14K)
    if (threadIdx.x == 0) {
        const uint8_t* mask = active_mask + (size_t)b * D_ffn;
        int* indices = active_indices + (size_t)b * D_ffn;
        int count = 0;

        for (int i = 0; i < D_ffn; i++) {
            if (mask[i]) {
                indices[count++] = i;
            }
        }
        active_count[b] = count;
    }
}

// ============================================================================
// HOST API
// ============================================================================

/**
 * @brief Run gate prediction to determine which MLP neurons are active.
 *
 * @param d_input          Input activations [B, D] FP16
 * @param d_gate_weights   Gate weights [D_ffn, D] FP16
 * @param d_gate_bias      Gate bias [D_ffn] FP16
 * @param d_active_mask    [out] Active mask [B, D_ffn] uint8
 * @param d_active_indices [out] Compact active indices [B, D_ffn] int
 * @param d_active_count   [out] Active count [B] int
 * @param d_gate_output    [out] Raw gate values [B, D_ffn] float (optional, for debugging)
 * @param B                Batch size
 * @param D                Input dim
 * @param D_ffn            FFN dim
 * @param threshold        Gate threshold
 * @param stream           CUDA stream
 * @return float           Elapsed time in ms
 */
extern "C" float predict_active_neurons(
    const half* d_input,
    const half* d_gate_weights,
    const half* d_gate_bias,
    uint8_t* d_active_mask,
    int* d_active_indices,
    int* d_active_count,
    float* d_gate_output,
    int B,
    int D,
    int D_ffn,
    float threshold,
    cudaStream_t stream
) {
    CudaTimer timer;
    timer.start(stream);

    // Reset counters
    CUDA_CHECK(cudaMemsetAsync(d_active_count, 0, B * sizeof(int), stream));

    // Gate prediction
    dim3 grid1(B, cdiv(D_ffn, GATE_BLOCK_SIZE));
    dim3 block1(GATE_BLOCK_SIZE);

    gate_predict_kernel<<<grid1, block1, 0, stream>>>(
        d_input, d_gate_weights, d_gate_bias,
        d_gate_output, d_active_mask, d_active_count,
        B, D, D_ffn, threshold
    );
    CUDA_CHECK_KERNEL();

    // Reset counters for compact pass
    CUDA_CHECK(cudaMemsetAsync(d_active_count, 0, B * sizeof(int), stream));

    // Compact indices
    compact_active_indices_kernel<<<B, 1, 0, stream>>>(
        d_active_mask, d_active_indices, d_active_count, B, D_ffn
    );
    CUDA_CHECK_KERNEL();

    timer.stop(stream);
    return timer.elapsed();
}

/**
 * @brief Get sparsity statistics for the current batch.
 *
 * @param d_active_count  Active counts [B]
 * @param B               Batch size
 * @param D_ffn           FFN dimension (total neurons)
 * @param stream          CUDA stream
 * @return float          Average sparsity percentage (0-100)
 */
extern "C" float get_sparsity_stats(
    const int* d_active_count,
    int B,
    int D_ffn,
    cudaStream_t stream
) {
    std::vector<int> h_counts(B);
    cuda_copy_d2h(h_counts.data(), d_active_count, B);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float total_active = 0.0f;
    for (int b = 0; b < B; b++) {
        total_active += (float)h_counts[b];
    }
    float avg_active = total_active / (float)B;
    float sparsity = (1.0f - avg_active / (float)D_ffn) * 100.0f;

    return sparsity;
}
