/**
 * @file kv_encode.cu
 * @brief PHANTOM CORE — KV-Cache Autoencoder Encode Kernel
 *
 * Innovation 3: Neural Cache — Learned KV-Cache Compression
 *
 * Compresses KV-cache entries from dimension D to D/8 using a trained
 * 3-layer autoencoder's encoder path:
 *   Encoder: Linear(D, D//4) -> GELU -> Linear(D//4, D//8)
 *
 * The encoder runs on every new KV-cache entry as it's generated,
 * storing compressed representations that use 8x less VRAM.
 *
 * Copyright (c) 2025 PHANTOM CORE Project
 */

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

constexpr int KV_ENCODE_BLOCK = 256;

// ============================================================================
// DEVICE FUNCTIONS
// ============================================================================

/**
 * @brief GELU activation function (exact formulation).
 */
__device__ inline float gelu_exact(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// ============================================================================
// KERNEL 1: FUSED ENCODER (LINEAR1 + GELU + LINEAR2)
// ============================================================================

/**
 * @brief Fused KV-cache encoder: compresses a batch of KV vectors from D to D/8.
 *
 * Performs:
 *   hidden = GELU(input @ W1^T + b1)     [B*S, D] -> [B*S, D/4]
 *   output = hidden @ W2^T + b2          [B*S, D/4] -> [B*S, D/8]
 *
 * @param input      KV vectors to compress [B*S, D] in FP16
 * @param w1         Encoder layer 1 weights [D/4, D] in FP16
 * @param b1         Encoder layer 1 bias [D/4] in FP16
 * @param w2         Encoder layer 2 weights [D/8, D/4] in FP16
 * @param b2         Encoder layer 2 bias [D/8] in FP16
 * @param output     Compressed KV vectors [B*S, D/8] in FP16
 * @param batch_seq  B * S (total number of vectors to encode)
 * @param D          Original dimension
 *
 * Grid: (batch_seq, cdiv(D/8, blockDim.x))
 * Block: (KV_ENCODE_BLOCK)
 */
__global__ void kv_encode_fused_kernel(
    const half* __restrict__ input,
    const half* __restrict__ w1,
    const half* __restrict__ b1,
    const half* __restrict__ w2,
    const half* __restrict__ b2,
    half* __restrict__ output,
    int batch_seq,
    int D
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= batch_seq) return;

    int D_4 = D / 4;
    int D_8 = D / 8;
    int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (out_idx >= D_8) return;

    const half* x = input + (size_t)vec_idx * D;

    // We need to compute the full hidden layer first for this output element
    // hidden[h] = GELU(Σ_d x[d] * w1[h,d] + b1[h])  for all h in [0, D/4)
    // output[out_idx] = Σ_h hidden[h] * w2[out_idx, h] + b2[out_idx]

    // Since D/4 can be large, we use shared memory to cache parts of the hidden layer
    extern __shared__ float s_hidden[];

    // Phase 1: Compute hidden = GELU(x @ W1^T + b1)
    // Each thread in the block computes a subset of hidden units
    for (int h = threadIdx.x; h < D_4; h += blockDim.x) {
        float acc = __half2float(b1[h]);
        for (int d = 0; d < D; d++) {
            acc += __half2float(x[d]) * __half2float(w1[(size_t)h * D + d]);
        }
        s_hidden[h] = gelu_exact(acc);
    }
    __syncthreads();

    // Phase 2: Compute output = hidden @ W2^T + b2
    float result = __half2float(b2[out_idx]);
    for (int h = 0; h < D_4; h++) {
        result += s_hidden[h] * __half2float(w2[(size_t)out_idx * D_4 + h]);
    }

    output[(size_t)vec_idx * D_8 + out_idx] = __float2half(result);
}

// ============================================================================
// KERNEL 2: BATCHED ENCODER (OPTIMIZED FOR LARGE BATCHES)
// ============================================================================

/**
 * @brief Optimized batch encoder using a two-pass approach.
 *        Pass 1: Compute all hidden activations [B*S, D/4]
 *        Pass 2: Compute all outputs [B*S, D/8]
 *
 * This version is more memory-efficient for large D by splitting the computation.
 *
 * @param input      [B*S, D] in FP16
 * @param w1         [D/4, D] in FP16
 * @param b1         [D/4] in FP16
 * @param hidden     [B*S, D/4] in FP16 — intermediate buffer
 * @param batch_seq  B * S
 * @param D          Original dimension
 */
__global__ void kv_encode_pass1_kernel(
    const half* __restrict__ input,
    const half* __restrict__ w1,
    const half* __restrict__ b1,
    half* __restrict__ hidden,
    int batch_seq,
    int D
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= batch_seq) return;

    int D_4 = D / 4;
    int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= D_4) return;

    const half* x = input + (size_t)vec_idx * D;

    float acc = __half2float(b1[h]);
    for (int d = 0; d < D; d++) {
        acc += __half2float(x[d]) * __half2float(w1[(size_t)h * D + d]);
    }

    hidden[(size_t)vec_idx * D_4 + h] = __float2half(gelu_exact(acc));
}

/**
 * @brief Pass 2: hidden -> compressed output
 *
 * @param hidden     [B*S, D/4] in FP16
 * @param w2         [D/8, D/4] in FP16
 * @param b2         [D/8] in FP16
 * @param output     [B*S, D/8] in FP16
 * @param batch_seq  B * S
 * @param D          Original dimension
 */
__global__ void kv_encode_pass2_kernel(
    const half* __restrict__ hidden,
    const half* __restrict__ w2,
    const half* __restrict__ b2,
    half* __restrict__ output,
    int batch_seq,
    int D
) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= batch_seq) return;

    int D_4 = D / 4;
    int D_8 = D / 8;
    int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (out_idx >= D_8) return;

    const half* h_vec = hidden + (size_t)vec_idx * D_4;

    float acc = __half2float(b2[out_idx]);
    for (int h = 0; h < D_4; h++) {
        acc += __half2float(h_vec[h]) * __half2float(w2[(size_t)out_idx * D_4 + h]);
    }

    output[(size_t)vec_idx * D_8 + out_idx] = __float2half(acc);
}

// ============================================================================
// HOST API
// ============================================================================

/**
 * @brief Encode (compress) KV-cache vectors using the trained autoencoder.
 *
 * @param d_input      Input KV vectors [batch_seq, D] in FP16
 * @param d_w1         Encoder weights layer 1 [D/4, D] in FP16
 * @param d_b1         Encoder bias layer 1 [D/4] in FP16
 * @param d_w2         Encoder weights layer 2 [D/8, D/4] in FP16
 * @param d_b2         Encoder bias layer 2 [D/8] in FP16
 * @param d_output     Compressed output [batch_seq, D/8] in FP16
 * @param batch_seq    Number of vectors to encode
 * @param D            Original dimension (e.g., 8192 for Llama-3 70B)
 * @param stream       CUDA stream
 * @return float       Elapsed time in ms
 */
extern "C" float kv_cache_encode(
    const half* d_input,
    const half* d_w1,
    const half* d_b1,
    const half* d_w2,
    const half* d_b2,
    half* d_output,
    int batch_seq,
    int D,
    cudaStream_t stream
) {
    CudaTimer timer;
    timer.start(stream);

    int D_4 = D / 4;
    int D_8 = D / 8;

    // Use two-pass approach for better memory access patterns
    half* d_hidden = cuda_malloc<half>((size_t)batch_seq * D_4);

    // Pass 1: input -> hidden (with GELU)
    dim3 grid1(batch_seq, cdiv(D_4, KV_ENCODE_BLOCK));
    dim3 block1(KV_ENCODE_BLOCK);
    kv_encode_pass1_kernel<<<grid1, block1, 0, stream>>>(
        d_input, d_w1, d_b1, d_hidden, batch_seq, D
    );
    CUDA_CHECK_KERNEL();

    // Pass 2: hidden -> compressed output
    dim3 grid2(batch_seq, cdiv(D_8, KV_ENCODE_BLOCK));
    dim3 block2(KV_ENCODE_BLOCK);
    kv_encode_pass2_kernel<<<grid2, block2, 0, stream>>>(
        d_hidden, d_w2, d_b2, d_output, batch_seq, D
    );
    CUDA_CHECK_KERNEL();

    cuda_free(d_hidden);

    timer.stop(stream);
    return timer.elapsed();
}
