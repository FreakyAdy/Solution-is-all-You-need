/**
 * @file gate_calibrate.cu
 * @brief PHANTOM CORE — Sparsity Gate Calibration
 *
 * Trains the linear gate predictors for Innovation 5 (Adaptive Compute Routing).
 *
 * For each MLP block, trains:
 *   gate = sigmoid(W_gate @ x + b_gate)
 *
 * to predict the neuron activation binary mask. Training uses binary
 * cross-entropy loss with the actual neuron activations as ground truth.
 *
 * Gate calibration determines:
 * - Whether sparse routing is beneficial per layer (F1 > 0.70 threshold)
 * - Optimal activation threshold per layer
 * - Expected sparsity percentage per layer
 *
 * Copyright (c) 2025 PHANTOM CORE Project
 */

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// ============================================================================
// CONFIGURATION
// ============================================================================

constexpr int CAL_BLOCK_SIZE = 256;
constexpr float CAL_LR = 1e-3f;
constexpr int CAL_EPOCHS = 50;
constexpr float CAL_F1_THRESHOLD = 0.70f;
constexpr float CAL_ACTIVATION_THRESHOLD = 0.01f;

// ============================================================================
// KERNEL: COMPUTE GROUND TRUTH ACTIVATION MASKS
// ============================================================================

/**
 * @brief Generate ground truth binary activation masks from MLP activations.
 *
 * A neuron is considered "active" if its absolute activation exceeds
 * CAL_ACTIVATION_THRESHOLD * max_activation_in_row.
 *
 * @param activations   MLP neuron activations [N_samples, D_ffn] FP32
 * @param masks         Output binary masks [N_samples, D_ffn] uint8
 * @param N_samples     Number of calibration samples
 * @param D_ffn         FFN intermediate dimension
 */
__global__ void compute_activation_masks_kernel(
    const float* __restrict__ activations,
    uint8_t* __restrict__ masks,
    int N_samples,
    int D_ffn
) {
    int sample = blockIdx.x;
    if (sample >= N_samples) return;

    int neuron = blockIdx.y * blockDim.x + threadIdx.x;
    if (neuron >= D_ffn) return;

    const float* act_row = activations + (size_t)sample * D_ffn;

    // Find row maximum for relative thresholding
    // Use shared memory reduction
    __shared__ float s_max[CAL_BLOCK_SIZE / WARP_SIZE + 1];
    float local_max = 0.0f;
    for (int d = threadIdx.x; d < D_ffn; d += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(act_row[d]));
    }
    local_max = block_reduce_max(local_max, s_max);
    __shared__ float row_max;
    if (threadIdx.x == 0) {
        row_max = fmaxf(local_max, 1e-10f);
    }
    __syncthreads();

    // Apply threshold
    float abs_val = fabsf(act_row[neuron]);
    masks[(size_t)sample * D_ffn + neuron] =
        (abs_val >= CAL_ACTIVATION_THRESHOLD * row_max) ? 1 : 0;
}

// ============================================================================
// KERNEL: GATE FORWARD + BCE LOSS
// ============================================================================

/**
 * @brief Compute gate predictions and binary cross-entropy loss.
 *
 * @param inputs         Input activations [N_samples, D] FP32
 * @param gate_w         Gate weights [D_ffn, D] FP32
 * @param gate_b         Gate bias [D_ffn] FP32
 * @param gt_masks       Ground truth masks [N_samples, D_ffn] uint8
 * @param predictions    Output predictions [N_samples, D_ffn] FP32
 * @param loss_per_neuron Per-neuron average BCE loss [D_ffn] FP32
 * @param N_samples      Number of samples
 * @param D              Input dimension
 * @param D_ffn          FFN dimension
 */
__global__ void gate_forward_bce_kernel(
    const float* __restrict__ inputs,
    const float* __restrict__ gate_w,
    const float* __restrict__ gate_b,
    const uint8_t* __restrict__ gt_masks,
    float* __restrict__ predictions,
    float* __restrict__ loss_per_neuron,
    int N_samples,
    int D,
    int D_ffn
) {
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= D_ffn) return;

    const float* w_row = gate_w + (size_t)neuron * D;
    float b = gate_b[neuron];
    float total_loss = 0.0f;

    for (int s = 0; s < N_samples; s++) {
        const float* x = inputs + (size_t)s * D;

        // Forward: sigmoid(w @ x + b)
        float acc = b;
        for (int d = 0; d < D; d++) {
            acc += x[d] * w_row[d];
        }
        float pred = 1.0f / (1.0f + expf(-acc));

        // Clamp for numerical stability
        pred = fmaxf(1e-7f, fminf(1.0f - 1e-7f, pred));

        predictions[(size_t)s * D_ffn + neuron] = pred;

        // BCE: -[y*log(p) + (1-y)*log(1-p)]
        float y = (float)gt_masks[(size_t)s * D_ffn + neuron];
        float bce = -(y * logf(pred) + (1.0f - y) * logf(1.0f - pred));
        total_loss += bce;
    }

    loss_per_neuron[neuron] = total_loss / (float)N_samples;
}

// ============================================================================
// KERNEL: GATE GRADIENT + SGD UPDATE
// ============================================================================

/**
 * @brief Compute gradients and update gate weights with SGD.
 *
 * @param inputs         Input activations [N_samples, D] FP32
 * @param predictions    Gate predictions [N_samples, D_ffn] FP32
 * @param gt_masks       Ground truth [N_samples, D_ffn] uint8
 * @param gate_w         Gate weights [D_ffn, D] FP32 (updated in place)
 * @param gate_b         Gate bias [D_ffn] FP32 (updated in place)
 * @param lr             Learning rate
 * @param N_samples      Number of samples
 * @param D              Input dim
 * @param D_ffn          FFN dim
 */
__global__ void gate_sgd_update_kernel(
    const float* __restrict__ inputs,
    const float* __restrict__ predictions,
    const uint8_t* __restrict__ gt_masks,
    float* __restrict__ gate_w,
    float* __restrict__ gate_b,
    float lr,
    int N_samples,
    int D,
    int D_ffn
) {
    int neuron = blockIdx.x;
    if (neuron >= D_ffn) return;

    int d = blockIdx.y * blockDim.x + threadIdx.x;

    // Compute average gradient for this neuron's weight[d]
    // dL/dw[d] = (1/N) * Σ_s (pred_s - y_s) * x_s[d]
    // dL/db = (1/N) * Σ_s (pred_s - y_s)
    float grad_w = 0.0f;
    float grad_b = 0.0f;

    for (int s = 0; s < N_samples; s++) {
        float error = predictions[(size_t)s * D_ffn + neuron]
                    - (float)gt_masks[(size_t)s * D_ffn + neuron];
        if (d < D) {
            grad_w += error * inputs[(size_t)s * D + d];
        }
        if (threadIdx.x == 0 && blockIdx.y == 0) {
            grad_b += error;
        }
    }

    grad_w /= (float)N_samples;
    grad_b /= (float)N_samples;

    // SGD update
    if (d < D) {
        gate_w[(size_t)neuron * D + d] -= lr * grad_w;
    }
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        gate_b[neuron] -= lr * grad_b;
    }
}

// ============================================================================
// HOST API
// ============================================================================

/**
 * @brief Calibrate sparsity gates for one MLP layer.
 *
 * @param d_inputs         Calibration input activations [N_samples, D] FP32
 * @param d_mlp_activations MLP neuron activations (ground truth) [N_samples, D_ffn] FP32
 * @param N_samples        Number of samples
 * @param D                Input dimension
 * @param D_ffn            FFN intermediate dimension
 * @param d_gate_w         [out] Trained gate weights [D_ffn, D] FP32
 * @param d_gate_b         [out] Trained gate bias [D_ffn] FP32
 * @param h_enabled        [out] Whether gate is enabled for this layer
 * @param h_f1_score       [out] Achieved F1 score
 * @param h_sparsity       [out] Average sparsity percentage
 * @param stream           CUDA stream
 */
extern "C" void calibrate_sparsity_gate(
    const float* d_inputs,
    const float* d_mlp_activations,
    int N_samples,
    int D,
    int D_ffn,
    float* d_gate_w,
    float* d_gate_b,
    bool* h_enabled,
    float* h_f1_score,
    float* h_sparsity,
    cudaStream_t stream
) {
    printf("[PHANTOM CORE] Calibrating sparsity gate: D=%d, D_ffn=%d, N=%d\n",
           D, D_ffn, N_samples);

    // Step 1: Compute ground truth activation masks
    uint8_t* d_gt_masks = cuda_malloc<uint8_t>((size_t)N_samples * D_ffn);
    dim3 mask_grid(N_samples, cdiv(D_ffn, CAL_BLOCK_SIZE));
    compute_activation_masks_kernel<<<mask_grid, CAL_BLOCK_SIZE, 0, stream>>>(
        d_mlp_activations, d_gt_masks, N_samples, D_ffn
    );
    CUDA_CHECK_KERNEL();

    // Compute baseline sparsity
    std::vector<uint8_t> h_masks((size_t)N_samples * D_ffn);
    cuda_copy_d2h(h_masks.data(), d_gt_masks, (size_t)N_samples * D_ffn);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    long active_total = 0;
    for (size_t i = 0; i < (size_t)N_samples * D_ffn; i++) {
        active_total += h_masks[i];
    }
    float baseline_sparsity = 1.0f - (float)active_total / ((float)N_samples * D_ffn);
    *h_sparsity = baseline_sparsity * 100.0f;
    printf("[PHANTOM CORE] Baseline sparsity: %.1f%%\n", baseline_sparsity * 100.0f);

    // Initialize gate weights (Xavier)
    CUDA_CHECK(cudaMemsetAsync(d_gate_w, 0, (size_t)D_ffn * D * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_gate_b, 0, (size_t)D_ffn * sizeof(float), stream));

    // Allocate prediction buffer
    float* d_predictions = cuda_malloc<float>((size_t)N_samples * D_ffn);
    float* d_loss = cuda_malloc<float>(D_ffn);

    // Training loop
    for (int epoch = 0; epoch < CAL_EPOCHS; epoch++) {
        // Forward + loss
        gate_forward_bce_kernel<<<cdiv(D_ffn, CAL_BLOCK_SIZE), CAL_BLOCK_SIZE, 0, stream>>>(
            d_inputs, d_gate_w, d_gate_b, d_gt_masks,
            d_predictions, d_loss, N_samples, D, D_ffn
        );
        CUDA_CHECK_KERNEL();

        // Gradient + update
        dim3 update_grid(D_ffn, cdiv(D, CAL_BLOCK_SIZE));
        gate_sgd_update_kernel<<<update_grid, CAL_BLOCK_SIZE, 0, stream>>>(
            d_inputs, d_predictions, d_gt_masks,
            d_gate_w, d_gate_b, CAL_LR, N_samples, D, D_ffn
        );
        CUDA_CHECK_KERNEL();

        if (epoch % 10 == 0 || epoch == CAL_EPOCHS - 1) {
            // Compute average loss
            std::vector<float> h_loss(D_ffn);
            cuda_copy_d2h(h_loss.data(), d_loss, D_ffn);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            float avg_loss = 0.0f;
            for (int i = 0; i < D_ffn; i++) avg_loss += h_loss[i];
            avg_loss /= D_ffn;
            printf("[PHANTOM CORE] Gate epoch %d/%d — BCE Loss: %.4f\n",
                   epoch + 1, CAL_EPOCHS, avg_loss);
        }
    }

    // Step 3: Evaluate F1 score
    // Get final predictions
    gate_forward_bce_kernel<<<cdiv(D_ffn, CAL_BLOCK_SIZE), CAL_BLOCK_SIZE, 0, stream>>>(
        d_inputs, d_gate_w, d_gate_b, d_gt_masks,
        d_predictions, d_loss, N_samples, D, D_ffn
    );
    CUDA_CHECK_KERNEL();

    std::vector<float> h_preds((size_t)N_samples * D_ffn);
    cuda_copy_d2h(h_preds.data(), d_predictions, (size_t)N_samples * D_ffn);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compute precision, recall, F1
    long tp = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < (size_t)N_samples * D_ffn; i++) {
        bool pred = h_preds[i] >= 0.5f;
        bool actual = h_masks[i] > 0;
        if (pred && actual) tp++;
        else if (pred && !actual) fp++;
        else if (!pred && actual) fn++;
    }

    float precision = (tp + fp > 0) ? (float)tp / (float)(tp + fp) : 0.0f;
    float recall = (tp + fn > 0) ? (float)tp / (float)(tp + fn) : 0.0f;
    float f1 = (precision + recall > 0) ? 2.0f * precision * recall / (precision + recall) : 0.0f;

    *h_f1_score = f1;
    *h_enabled = (f1 >= CAL_F1_THRESHOLD);

    printf("[PHANTOM CORE] Gate calibration complete:\n");
    printf("  Precision: %.3f, Recall: %.3f, F1: %.3f\n", precision, recall, f1);
    printf("  Sparse routing %s for this layer (F1 threshold: %.2f)\n",
           *h_enabled ? "ENABLED" : "DISABLED", CAL_F1_THRESHOLD);

    cuda_free(d_gt_masks);
    cuda_free(d_predictions);
    cuda_free(d_loss);
}
