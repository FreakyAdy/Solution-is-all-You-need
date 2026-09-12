/**
 * @file gqa_kernel.cu
 * @brief PHANTOM CORE — Grouped Query Attention (GQA) Fused Kernel
 *
 * Handles the GQA pattern used by Llama-3, Mistral, Qwen2, and most modern
 * LLMs where multiple query heads share a single KV head.
 *
 * GQA: H_q query heads share H_kv KV heads.
 * Ratio: num_groups = H_q / H_kv (e.g., 8 query heads per KV head)
 *
 * This kernel is a specialization of the FlashAttention kernel that
 * natively handles the head mapping without repeating K/V in memory.
 *
 * Copyright (c) 2025 PHANTOM CORE Project
 */

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

// ============================================================================
// CONFIGURATION
// ============================================================================

constexpr int GQA_TILE_KV = 64;
constexpr int GQA_THREADS = 128;

// ============================================================================
// KERNEL: GQA ATTENTION WITH NATIVE HEAD MAPPING
// ============================================================================

/**
 * @brief GQA attention kernel with implicit K/V head broadcasting.
 *
 * @param Q        Query [B, H_q, S_q, D] FP16
 * @param K        Key [B, H_kv, S_kv, D] FP16
 * @param V        Value [B, H_kv, S_kv, D] FP16
 * @param O        Output [B, H_q, S_q, D] FP16
 * @param B        Batch size
 * @param H_q      Number of query heads
 * @param H_kv     Number of KV heads
 * @param S_q      Query sequence length
 * @param S_kv     KV sequence length
 * @param D        Head dimension
 * @param scale    1/sqrt(D)
 * @param is_causal Causal mask flag
 *
 * Grid: (B * H_q, S_q)
 * Block: (GQA_THREADS)
 */
__global__ void gqa_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int B,
    int H_q,
    int H_kv,
    int S_q,
    int S_kv,
    int D,
    float scale,
    bool is_causal
) {
    int bh_q = blockIdx.x;
    int b = bh_q / H_q;
    int h_q = bh_q % H_q;
    int q_idx = blockIdx.y;

    if (b >= B || q_idx >= S_q) return;

    // Map query head to KV head (GQA grouping)
    int num_groups = H_q / H_kv;
    int h_kv = h_q / num_groups;

    // Pointer to this query
    const half* q_ptr = Q + (((size_t)b * H_q + h_q) * S_q + q_idx) * D;

    // Pointers to K, V for the corresponding KV head
    const half* k_base = K + ((size_t)b * H_kv + h_kv) * S_kv * D;
    const half* v_base = V + ((size_t)b * H_kv + h_kv) * S_kv * D;

    // Load query into registers
    float q_vec[128];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        q_vec[d] = __half2float(q_ptr[d]);
    }

    // Full query vector needed for dot products — sync all threads
    __shared__ float s_query[128];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        s_query[d] = __half2float(q_ptr[d]);
    }
    __syncthreads();

    // Online softmax accumulators (per thread — each thread handles all KV positions)
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float o_acc[128];
    for (int d = 0; d < D && d < 128; d++) {
        o_acc[d] = 0.0f;
    }

    // Process only if this is thread 0 (single-thread per query for correctness)
    // Production version would parallelize across KV tiles within a warp
    if (threadIdx.x == 0) {
        int kv_end = is_causal ? min(S_kv, q_idx + 1) : S_kv;

        for (int kv_idx = 0; kv_idx < kv_end; kv_idx++) {
            const half* k_ptr = k_base + (size_t)kv_idx * D;
            const half* v_ptr = v_base + (size_t)kv_idx * D;

            // Dot product Q · K
            float score = 0.0f;
            for (int d = 0; d < D; d++) {
                score += s_query[d] * __half2float(k_ptr[d]);
            }
            score *= scale;

            // Online softmax
            float prev_max = row_max;
            row_max = fmaxf(row_max, score);
            float rescale = expf(prev_max - row_max);
            row_sum = row_sum * rescale + expf(score - row_max);

            float attn_weight = expf(score - row_max);
            for (int d = 0; d < D; d++) {
                o_acc[d] = o_acc[d] * rescale + attn_weight * __half2float(v_ptr[d]);
            }
        }

        // Write output
        half* o_ptr = O + (((size_t)b * H_q + h_q) * S_q + q_idx) * D;
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        for (int d = 0; d < D; d++) {
            o_ptr[d] = __float2half(o_acc[d] * inv_sum);
        }
    }
}

// ============================================================================
// KERNEL: ROPE POSITIONAL ENCODING (APPLIED IN-PLACE)
// ============================================================================

/**
 * @brief Apply Rotary Position Embedding (RoPE) to Q and K tensors.
 *
 * RoPE formula for dimension pair (2i, 2i+1):
 *   q_rotated[2i]   = q[2i] * cos(θ_i) - q[2i+1] * sin(θ_i)
 *   q_rotated[2i+1] = q[2i] * sin(θ_i) + q[2i+1] * cos(θ_i)
 *
 * where θ_i = position / (10000^(2i/D))
 *
 * @param qk       Q or K tensor [B, H, S, D] in FP16 (modified in place)
 * @param positions Position indices [B, S]
 * @param B        Batch
 * @param H        Heads
 * @param S        Sequence length
 * @param D        Head dimension
 * @param base     RoPE base frequency (default 10000.0)
 */
__global__ void apply_rope_kernel(
    half* __restrict__ qk,
    const int* __restrict__ positions,
    int B,
    int H,
    int S,
    int D,
    float base
) {
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int s = blockIdx.y;

    if (b >= B || s >= S) return;

    int pair_idx = threadIdx.x; // Each thread handles one dimension pair
    if (pair_idx >= D / 2) return;

    int pos = positions[b * S + s];
    float theta = (float)pos / powf(base, 2.0f * (float)pair_idx / (float)D);
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    half* vec = qk + (((size_t)b * H + h) * S + s) * D;

    float x0 = __half2float(vec[2 * pair_idx]);
    float x1 = __half2float(vec[2 * pair_idx + 1]);

    vec[2 * pair_idx]     = __float2half(x0 * cos_theta - x1 * sin_theta);
    vec[2 * pair_idx + 1] = __float2half(x0 * sin_theta + x1 * cos_theta);
}

// ============================================================================
// HOST API
// ============================================================================

/**
 * @brief Run GQA attention.
 *
 * @param d_Q, d_K, d_V  Input tensors
 * @param d_O             Output tensor
 * @param B, H_q, H_kv, S_q, S_kv, D  Dimensions
 * @param is_causal       Causal mask
 * @param stream          CUDA stream
 * @return float          Elapsed ms
 */
extern "C" float gqa_attention(
    const half* d_Q,
    const half* d_K,
    const half* d_V,
    half* d_O,
    int B,
    int H_q,
    int H_kv,
    int S_q,
    int S_kv,
    int D,
    bool is_causal,
    cudaStream_t stream
) {
    CudaTimer timer;
    timer.start(stream);

    float scale = 1.0f / sqrtf((float)D);

    dim3 grid(B * H_q, S_q);
    dim3 block(GQA_THREADS);

    gqa_attention_kernel<<<grid, block, 0, stream>>>(
        d_Q, d_K, d_V, d_O,
        B, H_q, H_kv, S_q, S_kv, D, scale, is_causal
    );
    CUDA_CHECK_KERNEL();

    timer.stop(stream);
    return timer.elapsed();
}

/**
 * @brief Apply RoPE to Q and K tensors.
 *
 * @param d_qk       Q or K tensor [B, H, S, D] FP16 (in-place)
 * @param d_positions Position indices [B, S]
 * @param B, H, S, D Dimensions
 * @param base       RoPE base frequency
 * @param stream     CUDA stream
 * @return float     Elapsed ms
 */
extern "C" float apply_rope(
    half* d_qk,
    const int* d_positions,
    int B,
    int H,
    int S,
    int D,
    float base,
    cudaStream_t stream
) {
    CudaTimer timer;
    timer.start(stream);

    dim3 grid(B * H, S);
    dim3 block(D / 2);

    apply_rope_kernel<<<grid, block, 0, stream>>>(
        d_qk, d_positions, B, H, S, D, base
    );
    CUDA_CHECK_KERNEL();

    timer.stop(stream);
    return timer.elapsed();
}
