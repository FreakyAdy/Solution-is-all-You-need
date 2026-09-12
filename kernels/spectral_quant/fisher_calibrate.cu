/**
 * @file fisher_calibrate.cu
 * @brief PHANTOM CORE — Fisher Information-Based K Selection
 *
 * Innovation 2 Calibration: Determines the optimal number of DCT coefficients
 * (K) to retain per row of each weight matrix, using Fisher Information as
 * a per-layer importance weight.
 *
 * Fisher Information Approximation:
 * ==================================
 * For weight w_ij, the Fisher Information is approximated as:
 *   F(w_ij) ≈ E[(∂L/∂w_ij)²] ≈ (1/S) * Σ_{s} (grad_s(w_ij))²
 *
 * where S is the number of calibration samples.
 *
 * The Fisher Information tells us how "important" each weight is for the model's
 * output. We use it to allocate the K budget: rows with higher Fisher trace
 * get more coefficients, rows with lower Fisher trace can tolerate more compression.
 *
 * K Selection Algorithm:
 * =====================
 * 1. Compute per-row Fisher trace F_row[m] = Σ_n F(w_{m,n})
 * 2. Compute DCT of each row
 * 3. For each row, sort DCT coefficients by |X[k]| * importance_weight
 * 4. Find minimum K such that spectral reconstruction error < threshold
 * 5. Apply Fisher-weighted budget: K[m] = K_base * (F_row[m] / F_mean)^alpha
 *    where alpha controls the sensitivity (default: 0.5)
 *
 * Copyright (c) 2025 PHANTOM CORE Project
 */

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// ============================================================================
// CONFIGURATION
// ============================================================================

constexpr int FISHER_BLOCK_SIZE = 256;

/// Default error threshold: retain enough coefficients for <0.5% perplexity increase
constexpr float DEFAULT_ERROR_THRESHOLD = 0.005f;

/// Fisher sensitivity exponent
constexpr float FISHER_ALPHA = 0.5f;

/// Minimum K (always keep at least this many coefficients per row)
constexpr int MIN_K_PER_ROW = 8;

/// Maximum K (cap to avoid diminishing returns)
constexpr float MAX_K_FRACTION = 0.5f;

// ============================================================================
// KERNEL 1: COMPUTE FISHER INFORMATION PER ROW
// ============================================================================

/**
 * @brief Compute diagonal Fisher Information approximation per row.
 *
 * @param gradients    Stacked gradient matrices [S x M x N] in FP16
 *                     S = number of calibration samples
 * @param fisher_rows  Output: per-row Fisher trace [M]
 * @param S            Number of calibration samples
 * @param M            Rows
 * @param N            Columns
 *
 * Grid: (M)
 * Block: (FISHER_BLOCK_SIZE)
 *
 * Each block processes one row, computing F_row[m] = (1/S) * Σ_s Σ_n grad²
 */
__global__ void compute_fisher_per_row_kernel(
    const half* __restrict__ gradients,
    float* __restrict__ fisher_rows,
    int S,
    int M,
    int N
) {
    int row = blockIdx.x;
    if (row >= M) return;

    __shared__ float s_reduce[FISHER_BLOCK_SIZE / WARP_SIZE + 1];

    float local_sum = 0.0f;

    // Sum squared gradients across all samples and columns
    for (int s = 0; s < S; s++) {
        const half* grad_row = gradients + ((size_t)s * M + row) * N;
        for (int n = threadIdx.x; n < N; n += blockDim.x) {
            float g = __half2float(grad_row[n]);
            local_sum += g * g;
        }
    }

    // Block reduce
    float total = block_reduce_sum(local_sum, s_reduce);

    if (threadIdx.x == 0) {
        fisher_rows[row] = total / (float)S;
    }
}

// ============================================================================
// KERNEL 2: COMPUTE SPECTRAL ENERGY PER ROW
// ============================================================================

/**
 * @brief Compute the total spectral energy and cumulative energy of sorted
 *        DCT coefficients per row, to determine K.
 *
 * @param dct_coeffs     Full DCT coefficient matrix [M x N]
 * @param row_energies   Output: total energy per row [M]
 * @param M, N           Dimensions
 */
__global__ void compute_spectral_energy_kernel(
    const float* __restrict__ dct_coeffs,
    float* __restrict__ row_energies,
    int M,
    int N
) {
    int row = blockIdx.x;
    if (row >= M) return;

    __shared__ float s_reduce[FISHER_BLOCK_SIZE / WARP_SIZE + 1];

    const float* row_dct = dct_coeffs + (size_t)row * N;
    float local_energy = 0.0f;

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        float c = row_dct[n];
        local_energy += c * c;
    }

    float total = block_reduce_sum(local_energy, s_reduce);

    if (threadIdx.x == 0) {
        row_energies[row] = total;
    }
}

// ============================================================================
// HOST API: FISHER-CALIBRATED K SELECTION
// ============================================================================

/**
 * @brief Compute optimal K values per row using Fisher Information weighting.
 *
 * This is the main calibration entry point. Call this during the calibration
 * phase for each MLP weight matrix.
 *
 * @param d_weights       Device: weight matrix [M x N] in FP16
 * @param d_gradients     Device: gradient samples [S x M x N] in FP16
 * @param d_dct_coeffs    Device: pre-computed DCT coefficients [M x N] in FP32
 *                        (output from dct_type2_kernel in dct_compress.cu)
 * @param S               Number of calibration samples
 * @param M               Rows (output neurons)
 * @param N               Columns (input dimension)
 * @param error_threshold Target reconstruction error fraction (default 0.005)
 * @param h_k_per_row     Host output: K values per row [M] in uint16
 * @param h_K_max         Host output: maximum K across all rows
 * @param stream          CUDA stream
 */
extern "C" void fisher_calibrate_k(
    const half* d_weights,
    const half* d_gradients,
    const float* d_dct_coeffs,
    int S,
    int M,
    int N,
    float error_threshold,
    uint16_t* h_k_per_row,
    int* h_K_max,
    cudaStream_t stream
) {
    if (error_threshold <= 0.0f) {
        error_threshold = DEFAULT_ERROR_THRESHOLD;
    }

    printf("[PHANTOM CORE] Fisher calibration: M=%d, N=%d, S=%d, threshold=%.4f\n",
           M, N, S, error_threshold);

    // Step 1: Compute Fisher Information per row
    float* d_fisher_rows = cuda_malloc<float>(M);
    compute_fisher_per_row_kernel<<<M, FISHER_BLOCK_SIZE, 0, stream>>>(
        d_gradients, d_fisher_rows, S, M, N
    );
    CUDA_CHECK_KERNEL();

    // Step 2: Compute spectral energy per row
    float* d_row_energies = cuda_malloc<float>(M);
    compute_spectral_energy_kernel<<<M, FISHER_BLOCK_SIZE, 0, stream>>>(
        d_dct_coeffs, d_row_energies, M, N
    );
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy to host for K selection (sequential selection is fine for calibration)
    std::vector<float> h_fisher(M);
    std::vector<float> h_energies(M);
    cuda_copy_d2h(h_fisher.data(), d_fisher_rows, M);
    cuda_copy_d2h(h_energies.data(), d_row_energies, M);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Also need full DCT coefficients on host for sorting
    std::vector<float> h_dct((size_t)M * N);
    cuda_copy_d2h(h_dct.data(), d_dct_coeffs, (size_t)M * N);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Step 3: Compute Fisher-weighted K per row
    // First, find mean Fisher trace
    float fisher_mean = 0.0f;
    for (int m = 0; m < M; m++) {
        fisher_mean += h_fisher[m];
    }
    fisher_mean /= (float)M;
    if (fisher_mean < 1e-10f) fisher_mean = 1e-10f;

    int max_k_cap = (int)((float)N * MAX_K_FRACTION);
    int global_K_max = 0;

    for (int m = 0; m < M; m++) {
        // Sort DCT coefficients by magnitude (descending)
        std::vector<std::pair<float, int>> coeff_mag(N);
        for (int n = 0; n < N; n++) {
            coeff_mag[n] = {fabsf(h_dct[(size_t)m * N + n]), n};
        }
        std::sort(coeff_mag.begin(), coeff_mag.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Find K: minimum coefficients to retain (1 - threshold) fraction of energy
        float total_energy = h_energies[m];
        if (total_energy < 1e-10f) {
            h_k_per_row[m] = MIN_K_PER_ROW;
            global_K_max = std::max(global_K_max, MIN_K_PER_ROW);
            continue;
        }

        float target_energy = total_energy * (1.0f - error_threshold);
        float cumulative = 0.0f;
        int base_k = MIN_K_PER_ROW;

        for (int i = 0; i < N; i++) {
            cumulative += coeff_mag[i].first * coeff_mag[i].first;
            if (cumulative >= target_energy) {
                base_k = i + 1;
                break;
            }
        }

        // Apply Fisher weighting: important rows get more coefficients
        float fisher_weight = powf(h_fisher[m] / fisher_mean, FISHER_ALPHA);
        fisher_weight = fmaxf(0.5f, fminf(2.0f, fisher_weight)); // Clamp [0.5, 2.0]

        int weighted_k = (int)roundf((float)base_k * fisher_weight);
        weighted_k = std::max(weighted_k, MIN_K_PER_ROW);
        weighted_k = std::min(weighted_k, max_k_cap);

        h_k_per_row[m] = (uint16_t)weighted_k;
        global_K_max = std::max(global_K_max, weighted_k);
    }

    *h_K_max = global_K_max;

    // Compute statistics
    float k_mean = 0.0f;
    int k_min = N, k_max_val = 0;
    for (int m = 0; m < M; m++) {
        int k = (int)h_k_per_row[m];
        k_mean += (float)k;
        k_min = std::min(k_min, k);
        k_max_val = std::max(k_max_val, k);
    }
    k_mean /= (float)M;

    float compression_ratio = (float)N / k_mean;

    printf("[PHANTOM CORE] Fisher K calibration results:\n");
    printf("  K range: [%d, %d], mean K: %.1f\n", k_min, k_max_val, k_mean);
    printf("  Compression ratio: %.2fx\n", compression_ratio);
    printf("  Memory reduction: %.1f%%\n", (1.0f - 1.0f / compression_ratio) * 100.0f);

    cuda_free(d_fisher_rows);
    cuda_free(d_row_energies);
}

/**
 * @brief Simplified K calibration without gradients (energy-only).
 *
 * Use this when gradient samples are not available. Selects K based purely
 * on spectral energy distribution with a uniform budget.
 *
 * @param d_dct_coeffs    Device: DCT coefficients [M x N] in FP32
 * @param M, N            Dimensions
 * @param error_threshold Target energy retention fraction
 * @param h_k_per_row     Host output: K values per row [M]
 * @param h_K_max         Host output: max K
 * @param stream          CUDA stream
 */
extern "C" void energy_calibrate_k(
    const float* d_dct_coeffs,
    int M,
    int N,
    float error_threshold,
    uint16_t* h_k_per_row,
    int* h_K_max,
    cudaStream_t stream
) {
    if (error_threshold <= 0.0f) {
        error_threshold = DEFAULT_ERROR_THRESHOLD;
    }

    printf("[PHANTOM CORE] Energy-based K calibration: M=%d, N=%d, threshold=%.4f\n",
           M, N, error_threshold);

    // Copy DCT to host
    std::vector<float> h_dct((size_t)M * N);
    cuda_copy_d2h(h_dct.data(), d_dct_coeffs, (size_t)M * N);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int max_k_cap = (int)((float)N * MAX_K_FRACTION);
    int global_K_max = 0;

    for (int m = 0; m < M; m++) {
        // Compute total energy
        float total_energy = 0.0f;
        for (int n = 0; n < N; n++) {
            float c = h_dct[(size_t)m * N + n];
            total_energy += c * c;
        }

        if (total_energy < 1e-10f) {
            h_k_per_row[m] = MIN_K_PER_ROW;
            global_K_max = std::max(global_K_max, MIN_K_PER_ROW);
            continue;
        }

        // Sort by magnitude
        std::vector<float> mags(N);
        for (int n = 0; n < N; n++) {
            mags[n] = fabsf(h_dct[(size_t)m * N + n]);
        }
        std::sort(mags.begin(), mags.end(), std::greater<float>());

        // Find minimum K for target energy
        float target = total_energy * (1.0f - error_threshold);
        float cumulative = 0.0f;
        int k = MIN_K_PER_ROW;

        for (int i = 0; i < N; i++) {
            cumulative += mags[i] * mags[i];
            if (cumulative >= target) {
                k = std::max(i + 1, MIN_K_PER_ROW);
                break;
            }
        }

        k = std::min(k, max_k_cap);
        h_k_per_row[m] = (uint16_t)k;
        global_K_max = std::max(global_K_max, k);
    }

    *h_K_max = global_K_max;

    float k_mean = 0.0f;
    for (int m = 0; m < M; m++) k_mean += (float)h_k_per_row[m];
    k_mean /= (float)M;

    printf("[PHANTOM CORE] Energy K calibration: mean K=%.1f, max K=%d, ratio=%.2fx\n",
           k_mean, global_K_max, (float)N / k_mean);
}
