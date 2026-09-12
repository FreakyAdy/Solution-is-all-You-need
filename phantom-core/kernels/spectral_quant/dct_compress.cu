/**
 * @file dct_compress.cu
 * @brief PHANTOM CORE — Spectral Quantization: 1D DCT Weight Compression
 *
 * Innovation 2: Frequency-Domain Weight Compression
 *
 * Mathematical Foundation:
 * =======================
 * 1D Type-II DCT for a sequence x[0..N-1]:
 *
 *   X[k] = sqrt(2/N) * c(k) * Σ_{n=0}^{N-1} x[n] * cos(π(2n+1)k / 2N)
 *
 * where c(0) = 1/sqrt(2), c(k) = 1 for k > 0
 *
 * Inverse (Type-III DCT):
 *   x[n] = sqrt(2/N) * Σ_{k=0}^{N-1} c(k) * X[k] * cos(π(2n+1)k / 2N)
 *
 * CUDA Tiling Strategy:
 * =====================
 * For a weight matrix [M x N] (e.g., 4096 x 14336):
 * - Each row is processed independently (per-row DCT)
 * - Grid: (M, cdiv(N, TILE_SIZE)) — one block-row per matrix row
 * - Block: (TILE_SIZE) threads, each computing one DCT coefficient
 * - Shared memory: TILE_SIZE floats for the input tile
 * - Non-power-of-2 N: pad with zeros in shared memory
 *
 * FP8 Storage:
 * ============
 * After DCT, retain only top-K coefficients per row (by absolute magnitude).
 * Store as: FP8 E4M3 coefficient values + uint16 index mask.
 * K is determined by Fisher Information calibration (see fisher_calibrate.cu).
 *
 * Target Performance: <8ms for 4096x14336 on RTX 4050
 *
 * Copyright (c) 2025 PHANTOM CORE Project
 */

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Tile size for DCT computation — must be power of 2
constexpr int DCT_TILE_SIZE = 256;

/// Maximum row dimension we support (can be increased)
constexpr int MAX_ROW_DIM = 65536;

/// Number of threads per block for the top-K selection kernel
constexpr int TOPK_BLOCK_SIZE = 256;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * @brief Compressed row storage for spectrally quantized weights.
 *
 * Each row of the weight matrix is compressed into:
 *   - k_count: number of retained DCT coefficients
 *   - coefficients: FP8 E4M3 values of the retained coefficients
 *   - indices: uint16 indices of the retained coefficients
 *   - scale: per-row FP32 scale factor for coefficient de-normalization
 */
struct SpectralCompressedRow {
    uint16_t k_count;       ///< Number of retained coefficients
    float scale;            ///< Per-row scale factor (max absolute DCT value)
    // Followed in memory by:
    // fp8_e4m3_t coefficients[k_count]
    // uint16_t   indices[k_count]
};

/**
 * @brief Metadata for a spectrally compressed weight matrix.
 */
struct SpectralCompressedMatrix {
    uint32_t M;             ///< Number of rows (output neurons)
    uint32_t N;             ///< Original number of columns
    uint32_t total_coeffs;  ///< Total retained coefficients across all rows
    float compression_ratio;///< Original size / compressed size
    // Followed by M x SpectralCompressedRow entries + packed coefficient data
};

// ============================================================================
// KERNEL 1: ROW-WISE 1D TYPE-II DCT
// ============================================================================

/**
 * @brief Compute 1D Type-II DCT for each row of a weight matrix.
 *
 * @param input     Input weight matrix [M x N] in FP16 on device
 * @param output    Output DCT coefficients [M x N] in FP32 on device
 * @param M         Number of rows
 * @param N         Number of columns (row length)
 *
 * Grid: (M, cdiv(N, blockDim.x))
 * Block: (DCT_TILE_SIZE)
 *
 * Each thread computes one DCT coefficient X[k] by accumulating the sum
 * over all input elements. For large N, we tile the input and accumulate
 * across tiles.
 */
__global__ void dct_type2_kernel(
    const half* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    // Row index
    int row = blockIdx.x;
    if (row >= M) return;

    // DCT coefficient index this thread computes
    int k = blockIdx.y * blockDim.x + threadIdx.x;
    if (k >= N) return;

    const half* row_input = input + (size_t)row * N;

    // Accumulate DCT sum: X[k] = Σ x[n] * cos(π(2n+1)k / 2N)
    float sum = 0.0f;
    float freq = PHANTOM_PI * (float)k / (2.0f * (float)N);

    // Process input in tiles for cache efficiency
    for (int tile_start = 0; tile_start < N; tile_start += DCT_TILE_SIZE) {
        // Shared memory for input tile
        __shared__ float s_tile[DCT_TILE_SIZE];

        int local_idx = threadIdx.x;
        int global_idx = tile_start + local_idx;

        // Load tile into shared memory (zero-pad if beyond N)
        if (global_idx < N) {
            s_tile[local_idx] = __half2float(row_input[global_idx]);
        } else {
            s_tile[local_idx] = 0.0f;
        }
        __syncthreads();

        // Accumulate contribution from this tile
        int tile_end = min(DCT_TILE_SIZE, N - tile_start);
        for (int i = 0; i < tile_end; i++) {
            int n = tile_start + i;
            float cos_val = cosf(freq * (2.0f * (float)n + 1.0f));
            sum += s_tile[i] * cos_val;
        }
        __syncthreads();
    }

    // Normalization factor: sqrt(2/N) * c(k)
    float norm = sqrtf(2.0f / (float)N);
    if (k == 0) {
        norm *= (1.0f / sqrtf(2.0f)); // c(0) = 1/sqrt(2)
    }

    output[(size_t)row * N + k] = norm * sum;
}

// ============================================================================
// KERNEL 2: TOP-K COEFFICIENT SELECTION
// ============================================================================

/**
 * @brief For each row, find the top-K DCT coefficients by absolute magnitude
 *        and write them to the compressed output.
 *
 * @param dct_coeffs   Full DCT coefficient matrix [M x N] in FP32
 * @param out_values   Output: FP8 coefficient values [M x K_max]
 * @param out_indices  Output: uint16 coefficient indices [M x K_max]
 * @param out_scales   Output: per-row FP32 scale factors [M]
 * @param out_k_counts Output: actual K per row [M]
 * @param k_per_row    Array of K values per row (from Fisher calibration) [M]
 * @param M            Number of rows
 * @param N            Row length
 *
 * Grid: (M)
 * Block: (TOPK_BLOCK_SIZE)
 *
 * Strategy: Each block processes one row. We use a parallel reduction to find
 * the K-th largest absolute value (threshold), then select all coefficients
 * above that threshold.
 */
__global__ void topk_select_kernel(
    const float* __restrict__ dct_coeffs,
    fp8_e4m3_t* __restrict__ out_values,
    uint16_t* __restrict__ out_indices,
    float* __restrict__ out_scales,
    uint16_t* __restrict__ out_k_counts,
    const uint16_t* __restrict__ k_per_row,
    int M,
    int N,
    int K_max
) {
    int row = blockIdx.x;
    if (row >= M) return;

    const float* row_coeffs = dct_coeffs + (size_t)row * N;
    int K = (int)k_per_row[row];
    K = min(K, K_max);
    K = min(K, N);

    // Phase 1: Find row-wise maximum absolute value (for scale normalization)
    __shared__ float s_reduce[TOPK_BLOCK_SIZE / WARP_SIZE + 1];
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(row_coeffs[i]));
    }
    local_max = block_reduce_max(local_max, s_reduce);
    __syncthreads();

    __shared__ float row_scale;
    if (threadIdx.x == 0) {
        row_scale = (local_max > 0.0f) ? local_max : 1.0f;
        out_scales[row] = row_scale;
    }
    __syncthreads();

    // Phase 2: Find the K-th largest absolute value using iterative thresholding
    // We use a histogram-based approach: binary search for the threshold
    __shared__ float threshold;
    __shared__ int count_above;

    if (threadIdx.x == 0) {
        // Simple iterative approach: start from max and lower until we have K
        // For production, use radix-select; this is a correct initial approach
        float lo = 0.0f, hi = row_scale;

        for (int iter = 0; iter < 32; iter++) { // 32 iterations of binary search
            float mid = (lo + hi) * 0.5f;
            int cnt = 0;
            for (int i = 0; i < N; i++) {
                if (fabsf(row_coeffs[i]) >= mid) cnt++;
            }
            if (cnt > K) {
                lo = mid;
            } else if (cnt < K) {
                hi = mid;
            } else {
                lo = mid;
                break;
            }
        }
        threshold = lo;

        // Count how many are above threshold
        int cnt = 0;
        for (int i = 0; i < N && cnt < K; i++) {
            if (fabsf(row_coeffs[i]) >= threshold) cnt++;
        }
        count_above = cnt;
    }
    __syncthreads();

    // Phase 3: Write the selected coefficients
    fp8_e4m3_t* row_values = out_values + (size_t)row * K_max;
    uint16_t* row_indices = out_indices + (size_t)row * K_max;

    if (threadIdx.x == 0) {
        int write_idx = 0;
        float inv_scale = 1.0f / row_scale;

        for (int i = 0; i < N && write_idx < K; i++) {
            float abs_val = fabsf(row_coeffs[i]);
            if (abs_val >= threshold) {
                // Normalize to [-1, 1] range for FP8 encoding
                float normalized = row_coeffs[i] * inv_scale;
                // Scale to FP8 range (±448 max, but we use ±1 normalized)
                row_values[write_idx] = float_to_fp8_e4m3(normalized);
                row_indices[write_idx] = (uint16_t)i;
                write_idx++;
            }
        }
        out_k_counts[row] = (uint16_t)write_idx;

        // Zero-fill remaining slots
        for (int i = write_idx; i < K_max; i++) {
            row_values[i] = fp8_e4m3_t(0);
            row_indices[i] = 0;
        }
    }
}

// ============================================================================
// KERNEL 3: INVERSE DCT (TYPE-III) — FULL ROW RECONSTRUCTION
// ============================================================================

/**
 * @brief Reconstruct weight matrix row from compressed spectral coefficients
 *        using inverse DCT (Type-III).
 *
 * @param values    FP8 coefficient values [M x K_max]
 * @param indices   Coefficient indices [M x K_max]
 * @param scales    Per-row scale factors [M]
 * @param k_counts  Actual K per row [M]
 * @param output    Reconstructed weight matrix [M x N] in FP16
 * @param M         Number of rows
 * @param N         Column dimension
 * @param K_max     Maximum K across all rows
 *
 * Grid: (M, cdiv(N, blockDim.x))
 * Block: (DCT_TILE_SIZE)
 *
 * Each thread reconstructs one output element x[n] by summing over the
 * sparse set of retained DCT coefficients.
 */
__global__ void idct_reconstruct_kernel(
    const fp8_e4m3_t* __restrict__ values,
    const uint16_t* __restrict__ indices,
    const float* __restrict__ scales,
    const uint16_t* __restrict__ k_counts,
    half* __restrict__ output,
    int M,
    int N,
    int K_max
) {
    int row = blockIdx.x;
    if (row >= M) return;

    int n = blockIdx.y * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const fp8_e4m3_t* row_values = values + (size_t)row * K_max;
    const uint16_t* row_indices = indices + (size_t)row * K_max;
    float scale = scales[row];
    int K = (int)k_counts[row];

    // iDCT: x[n] = sqrt(2/N) * Σ_{k in retained} c(k) * X[k] * cos(π(2n+1)k / 2N)
    float sum = 0.0f;
    float norm = sqrtf(2.0f / (float)N);

    for (int i = 0; i < K; i++) {
        int k = (int)row_indices[i];
        float coeff = fp8_e4m3_to_float(row_values[i]) * scale; // De-normalize

        float c_k = (k == 0) ? (1.0f / sqrtf(2.0f)) : 1.0f;
        float cos_val = cosf(PHANTOM_PI * (float)k * (2.0f * (float)n + 1.0f) / (2.0f * (float)N));

        sum += c_k * coeff * cos_val;
    }

    output[(size_t)row * N + n] = __float2half(norm * sum);
}

// ============================================================================
// HOST API: SPECTRAL COMPRESSION
// ============================================================================

/**
 * @brief Compress a weight matrix using Spectral Quantization (1D DCT + top-K + FP8).
 *
 * @param d_weights     Device pointer to input weight matrix [M x N] in FP16
 * @param M             Number of rows (output neurons)
 * @param N             Number of columns (input dimension)
 * @param d_k_per_row   Device pointer to K values per row [M] (from Fisher calibration)
 * @param K_max         Maximum K value across all rows
 * @param d_out_values  [out] Device pointer for FP8 coefficient values [M x K_max]
 * @param d_out_indices [out] Device pointer for uint16 coefficient indices [M x K_max]
 * @param d_out_scales  [out] Device pointer for per-row scale factors [M]
 * @param d_out_k_counts[out] Device pointer for actual K per row [M]
 * @param stream        CUDA stream for async execution
 *
 * @return float Time elapsed in milliseconds
 */
extern "C" float spectral_compress(
    const half* d_weights,
    int M,
    int N,
    const uint16_t* d_k_per_row,
    int K_max,
    fp8_e4m3_t* d_out_values,
    uint16_t* d_out_indices,
    float* d_out_scales,
    uint16_t* d_out_k_counts,
    cudaStream_t stream
) {
    CudaTimer timer;
    timer.start(stream);

    // Step 1: Compute full DCT of the weight matrix
    float* d_dct_full = cuda_malloc<float>((size_t)M * N);

    dim3 dct_grid(M, cdiv(N, DCT_TILE_SIZE));
    dim3 dct_block(DCT_TILE_SIZE);

    dct_type2_kernel<<<dct_grid, dct_block, 0, stream>>>(
        d_weights, d_dct_full, M, N
    );
    CUDA_CHECK_KERNEL();

    // Step 2: Select top-K coefficients per row
    dim3 topk_grid(M);
    dim3 topk_block(TOPK_BLOCK_SIZE);

    topk_select_kernel<<<topk_grid, topk_block, 0, stream>>>(
        d_dct_full, d_out_values, d_out_indices,
        d_out_scales, d_out_k_counts, d_k_per_row,
        M, N, K_max
    );
    CUDA_CHECK_KERNEL();

    // Free intermediate DCT buffer
    cuda_free(d_dct_full);

    timer.stop(stream);
    return timer.elapsed();
}

/**
 * @brief Decompress (reconstruct) a weight matrix from spectral coefficients.
 *
 * @param d_values      FP8 coefficient values [M x K_max]
 * @param d_indices     Coefficient indices [M x K_max]
 * @param d_scales      Per-row scale factors [M]
 * @param d_k_counts    Actual K per row [M]
 * @param d_output      [out] Reconstructed weight matrix [M x N] in FP16
 * @param M             Number of rows
 * @param N             Number of columns
 * @param K_max         Maximum K
 * @param stream        CUDA stream
 *
 * @return float Time elapsed in milliseconds
 */
extern "C" float spectral_decompress(
    const fp8_e4m3_t* d_values,
    const uint16_t* d_indices,
    const float* d_scales,
    const uint16_t* d_k_counts,
    half* d_output,
    int M,
    int N,
    int K_max,
    cudaStream_t stream
) {
    CudaTimer timer;
    timer.start(stream);

    dim3 grid(M, cdiv(N, DCT_TILE_SIZE));
    dim3 block(DCT_TILE_SIZE);

    idct_reconstruct_kernel<<<grid, block, 0, stream>>>(
        d_values, d_indices, d_scales, d_k_counts,
        d_output, M, N, K_max
    );
    CUDA_CHECK_KERNEL();

    timer.stop(stream);
    return timer.elapsed();
}

// ============================================================================
// HOST CORRECTNESS TEST — CPU REFERENCE IMPLEMENTATION
// ============================================================================

/**
 * @brief CPU reference implementation of 1D Type-II DCT for correctness testing.
 *
 * @param input   Input row [N]
 * @param output  Output DCT coefficients [N]
 * @param N       Length
 */
void cpu_dct_type2(const float* input, float* output, int N) {
    float norm = sqrtf(2.0f / (float)N);
    for (int k = 0; k < N; k++) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            sum += input[n] * cosf(PHANTOM_PI * (float)k * (2.0f * (float)n + 1.0f) / (2.0f * (float)N));
        }
        float c_k = (k == 0) ? (1.0f / sqrtf(2.0f)) : 1.0f;
        output[k] = norm * c_k * sum;
    }
}

/**
 * @brief CPU reference implementation of 1D Type-III iDCT for correctness testing.
 */
void cpu_idct_type3(const float* input, float* output, int N) {
    float norm = sqrtf(2.0f / (float)N);
    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            float c_k = (k == 0) ? (1.0f / sqrtf(2.0f)) : 1.0f;
            sum += c_k * input[k] * cosf(PHANTOM_PI * (float)k * (2.0f * (float)n + 1.0f) / (2.0f * (float)N));
        }
        output[n] = norm * sum;
    }
}

/**
 * @brief Self-test: verify GPU DCT against CPU reference.
 *
 * @return true if the test passes, false otherwise.
 */
bool test_dct_correctness() {
    constexpr int TEST_M = 4;
    constexpr int TEST_N = 128;

    printf("[PHANTOM CORE] Running DCT correctness test (M=%d, N=%d)...\n", TEST_M, TEST_N);

    // Generate random test input
    std::vector<float> h_input_f32(TEST_M * TEST_N);
    std::vector<half> h_input_f16(TEST_M * TEST_N);
    for (int i = 0; i < TEST_M * TEST_N; i++) {
        h_input_f32[i] = ((float)(rand() % 10000) / 10000.0f - 0.5f) * 2.0f;
        h_input_f16[i] = __float2half(h_input_f32[i]);
    }

    // CPU reference DCT
    std::vector<float> h_ref_dct(TEST_M * TEST_N, 0.0f);
    for (int row = 0; row < TEST_M; row++) {
        cpu_dct_type2(h_input_f32.data() + row * TEST_N,
                      h_ref_dct.data() + row * TEST_N,
                      TEST_N);
    }

    // GPU DCT
    half* d_input = cuda_malloc<half>(TEST_M * TEST_N);
    float* d_output = cuda_malloc<float>(TEST_M * TEST_N);
    cuda_copy_h2d(d_input, h_input_f16.data(), TEST_M * TEST_N);

    dim3 grid(TEST_M, cdiv(TEST_N, DCT_TILE_SIZE));
    dim3 block(DCT_TILE_SIZE);
    dct_type2_kernel<<<grid, block>>>(d_input, d_output, TEST_M, TEST_N);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_dct(TEST_M * TEST_N);
    cuda_copy_d2h(h_gpu_dct.data(), d_output, TEST_M * TEST_N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compare
    float max_error = 0.0f;
    for (int i = 0; i < TEST_M * TEST_N; i++) {
        float err = fabsf(h_gpu_dct[i] - h_ref_dct[i]);
        max_error = fmaxf(max_error, err);
    }

    printf("[PHANTOM CORE] DCT max absolute error vs CPU reference: %.6f\n", max_error);
    bool passed = (max_error < 0.05f); // FP16 input introduces some error

    // CPU reference iDCT roundtrip test
    std::vector<float> h_roundtrip(TEST_N, 0.0f);
    cpu_idct_type3(h_ref_dct.data(), h_roundtrip.data(), TEST_N);
    float roundtrip_error = 0.0f;
    for (int i = 0; i < TEST_N; i++) {
        roundtrip_error = fmaxf(roundtrip_error, fabsf(h_roundtrip[i] - h_input_f32[i]));
    }
    printf("[PHANTOM CORE] CPU DCT→iDCT roundtrip max error: %.8f\n", roundtrip_error);
    passed = passed && (roundtrip_error < 1e-5f);

    cuda_free(d_input);
    cuda_free(d_output);

    printf("[PHANTOM CORE] DCT correctness test: %s\n", passed ? "PASSED ✓" : "FAILED ✗");
    return passed;
}
