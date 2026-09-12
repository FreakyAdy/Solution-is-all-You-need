/**
 * @file idct_reconstruct.cu
 * @brief PHANTOM CORE — Fused iDCT Reconstruction + GEMM Kernel
 *
 * Innovation 2 Extension: This kernel reconstructs weight rows from spectral
 * coefficients AND fuses the reconstruction with a downstream matrix-vector
 * multiply (GEMV), eliminating the need to materialize the full weight matrix.
 *
 * Fused Operation:
 *   y[m] = Σ_{n} W_reconstructed[m,n] * x[n]
 *        = Σ_{n} (iDCT(sparse_coeffs[m]))[n] * x[n]
 *
 * By interchanging summation order:
 *   y[m] = Σ_{i in retained_k} scale[m] * decode(coeff_i) * c(k_i) *
 *          Σ_{n} x[n] * cos(π * k_i * (2n+1) / 2N)
 *
 * The inner sum is the DCT of x evaluated at frequency k_i — we precompute
 * DCT(x) once and reuse across all rows.
 *
 * Copyright (c) 2025 PHANTOM CORE Project
 */

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

constexpr int IDCT_BLOCK_SIZE = 256;
constexpr int IDCT_TILE = 64;

// ============================================================================
// KERNEL 1: PRECOMPUTE DCT OF INPUT VECTOR
// ============================================================================

/**
 * @brief Compute DCT of the input activation vector x[N].
 *
 * @param x         Input activation vector [N] in FP16
 * @param dct_x     Output DCT coefficients [N] in FP32
 * @param N         Vector length
 *
 * Grid: cdiv(N, blockDim.x)
 * Block: IDCT_BLOCK_SIZE
 */
__global__ void compute_input_dct_kernel(
    const half* __restrict__ x,
    float* __restrict__ dct_x,
    int N
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    float sum = 0.0f;
    float freq = PHANTOM_PI * (float)k / (2.0f * (float)N);

    for (int n = 0; n < N; n++) {
        float x_n = __half2float(x[n]);
        float cos_val = cosf(freq * (2.0f * (float)n + 1.0f));
        sum += x_n * cos_val;
    }

    float norm = sqrtf(2.0f / (float)N);
    float c_k = (k == 0) ? (1.0f / sqrtf(2.0f)) : 1.0f;
    dct_x[k] = norm * c_k * sum;
}

// ============================================================================
// KERNEL 2: FUSED iDCT-GEMV — THE CORE INNOVATION KERNEL
// ============================================================================

/**
 * @brief Fused spectral reconstruction + matrix-vector multiply.
 *
 * Instead of reconstructing the full weight row and then doing a dot product,
 * we directly compute:
 *   y[m] = scale[m] * Σ_{i=0}^{K-1} decode(coeff_values[m,i]) * dct_x[coeff_indices[m,i]]
 *
 * This is the Parseval/Plancherel identity exploit: dot product in spatial domain
 * equals dot product in frequency domain (up to normalization).
 *
 * @param coeff_values    FP8 coefficient values [M x K_max]
 * @param coeff_indices   Coefficient indices [M x K_max]
 * @param scales          Per-row scale factors [M]
 * @param k_counts        Actual K per row [M]
 * @param dct_x           Precomputed DCT of input vector [N]
 * @param output          Output vector [M] in FP16
 * @param M               Number of output elements (rows)
 * @param N               Original column dimension
 * @param K_max           Maximum retained coefficients per row
 * @param bias            Optional bias vector [M] in FP16 (can be nullptr)
 *
 * Grid: cdiv(M, blockDim.x)
 * Block: IDCT_BLOCK_SIZE
 */
__global__ void fused_idct_gemv_kernel(
    const fp8_e4m3_t* __restrict__ coeff_values,
    const uint16_t* __restrict__ coeff_indices,
    const float* __restrict__ scales,
    const uint16_t* __restrict__ k_counts,
    const float* __restrict__ dct_x,
    half* __restrict__ output,
    int M,
    int N,
    int K_max,
    const half* __restrict__ bias
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    int K = (int)k_counts[m];
    float scale = scales[m];

    const fp8_e4m3_t* row_values = coeff_values + (size_t)m * K_max;
    const uint16_t* row_indices = coeff_indices + (size_t)m * K_max;

    // Accumulate dot product in frequency domain
    float acc = 0.0f;
    for (int i = 0; i < K; i++) {
        float coeff = fp8_e4m3_to_float(row_values[i]) * scale;
        int k_idx = (int)row_indices[i];
        acc += coeff * dct_x[k_idx];
    }

    // The frequency-domain dot product gives us the spatial-domain dot product
    // (Parseval's theorem for DCT-II/III with proper normalization)
    float result = acc;

    // Add bias if present
    if (bias != nullptr) {
        result += __half2float(bias[m]);
    }

    output[m] = __float2half(result);
}

// ============================================================================
// KERNEL 3: STANDARD iDCT RECONSTRUCTION (NON-FUSED)
// ============================================================================

/**
 * @brief Reconstruct a full weight matrix row from sparse spectral coefficients.
 *        Use this when you need the actual reconstructed weights (e.g., for
 *        debugging or when fusion is not applicable).
 *
 * @param coeff_values   FP8 coefficient values [M x K_max]
 * @param coeff_indices  Coefficient indices [M x K_max]
 * @param scales         Per-row scale factors [M]
 * @param k_counts       Actual K per row [M]
 * @param output         Reconstructed matrix [M x N] in FP16
 * @param M              Rows
 * @param N              Columns
 * @param K_max          Max K
 */
__global__ void idct_full_reconstruct_kernel(
    const fp8_e4m3_t* __restrict__ coeff_values,
    const uint16_t* __restrict__ coeff_indices,
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

    const fp8_e4m3_t* row_values = coeff_values + (size_t)row * K_max;
    const uint16_t* row_indices = coeff_indices + (size_t)row * K_max;
    float scale = scales[row];
    int K = (int)k_counts[row];

    float sum = 0.0f;
    float norm = sqrtf(2.0f / (float)N);

    for (int i = 0; i < K; i++) {
        int k = (int)row_indices[i];
        float coeff = fp8_e4m3_to_float(row_values[i]) * scale;
        float c_k = (k == 0) ? (1.0f / sqrtf(2.0f)) : 1.0f;
        float cos_val = cosf(PHANTOM_PI * (float)k * (2.0f * (float)n + 1.0f)
                             / (2.0f * (float)N));
        sum += c_k * coeff * cos_val;
    }

    output[(size_t)row * N + n] = __float2half(norm * sum);
}

// ============================================================================
// HOST API
// ============================================================================

/**
 * @brief Fused spectral-domain matrix-vector multiply.
 *
 * Computes y = W_spectral @ x + bias without materializing W.
 *
 * @param d_coeff_values   FP8 coefficients [M x K_max]
 * @param d_coeff_indices  Coefficient indices [M x K_max]
 * @param d_scales         Row scales [M]
 * @param d_k_counts       K per row [M]
 * @param d_x              Input activation vector [N] in FP16
 * @param d_output         Output vector [M] in FP16
 * @param d_bias           Bias vector [M] in FP16 (or nullptr)
 * @param M                Output dimension
 * @param N                Input dimension
 * @param K_max            Max retained coefficients
 * @param stream           CUDA stream
 * @return float           Elapsed time in ms
 */
extern "C" float fused_spectral_gemv(
    const fp8_e4m3_t* d_coeff_values,
    const uint16_t* d_coeff_indices,
    const float* d_scales,
    const uint16_t* d_k_counts,
    const half* d_x,
    half* d_output,
    const half* d_bias,
    int M,
    int N,
    int K_max,
    cudaStream_t stream
) {
    CudaTimer timer;
    timer.start(stream);

    // Step 1: Precompute DCT of input vector
    float* d_dct_x = cuda_malloc<float>(N);
    int dct_blocks = cdiv(N, IDCT_BLOCK_SIZE);
    compute_input_dct_kernel<<<dct_blocks, IDCT_BLOCK_SIZE, 0, stream>>>(
        d_x, d_dct_x, N
    );
    CUDA_CHECK_KERNEL();

    // Step 2: Fused iDCT-GEMV
    int gemv_blocks = cdiv(M, IDCT_BLOCK_SIZE);
    fused_idct_gemv_kernel<<<gemv_blocks, IDCT_BLOCK_SIZE, 0, stream>>>(
        d_coeff_values, d_coeff_indices, d_scales, d_k_counts,
        d_dct_x, d_output, M, N, K_max, d_bias
    );
    CUDA_CHECK_KERNEL();

    cuda_free(d_dct_x);

    timer.stop(stream);
    return timer.elapsed();
}

/**
 * @brief Reconstruct full weight matrix from spectral coefficients.
 *
 * @param d_coeff_values   FP8 coefficients [M x K_max]
 * @param d_coeff_indices  Coefficient indices [M x K_max]
 * @param d_scales         Row scales [M]
 * @param d_k_counts       K per row [M]
 * @param d_output         Reconstructed matrix [M x N] in FP16
 * @param M, N             Dimensions
 * @param K_max            Max K
 * @param stream           CUDA stream
 * @return float           Elapsed time in ms
 */
extern "C" float spectral_full_reconstruct(
    const fp8_e4m3_t* d_coeff_values,
    const uint16_t* d_coeff_indices,
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

    dim3 grid(M, cdiv(N, IDCT_BLOCK_SIZE));
    dim3 block(IDCT_BLOCK_SIZE);

    idct_full_reconstruct_kernel<<<grid, block, 0, stream>>>(
        d_coeff_values, d_coeff_indices, d_scales, d_k_counts,
        d_output, M, N, K_max
    );
    CUDA_CHECK_KERNEL();

    timer.stop(stream);
    return timer.elapsed();
}
