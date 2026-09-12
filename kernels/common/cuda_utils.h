/**
 * @file cuda_utils.h
 * @brief PHANTOM CORE — Shared CUDA Utilities
 *
 * Common macros, FP8 encoding/decoding, error checking, and timing utilities
 * shared across all PHANTOM CORE custom CUDA kernels.
 *
 * Copyright (c) 2025 PHANTOM CORE Project
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>

// ============================================================================
// ERROR CHECKING
// ============================================================================

/**
 * @brief CUDA API error checking macro. Terminates with diagnostic on failure.
 * @param call Any CUDA API call returning cudaError_t.
 */
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "[PHANTOM CORE CUDA ERROR] %s:%d — %s: %s\n",     \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/**
 * @brief Check for kernel launch errors (asynchronous). Call after kernel<<<>>>().
 */
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                        \
        cudaError_t err = cudaGetLastError();                                   \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "[PHANTOM CORE KERNEL ERROR] %s:%d — %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ============================================================================
// CONSTANTS
// ============================================================================

/// Warp size for all NVIDIA GPUs
constexpr int WARP_SIZE = 32;

/// Tile size for shared memory tiling (optimized for Ampere/Ada L1 cache)
constexpr int TILE_DIM = 32;

/// Maximum threads per block (conservative for register pressure)
constexpr int MAX_THREADS_PER_BLOCK = 256;

/// Block size for 1D kernels
constexpr int BLOCK_SIZE_1D = 256;

/// Shared memory bank count
constexpr int SHARED_BANKS = 32;

/// PI constant for DCT computations
constexpr float PHANTOM_PI = 3.14159265358979323846f;

// ============================================================================
// FP8 ENCODING (E4M3 FORMAT)
// ============================================================================

/**
 * @brief FP8 E4M3 representation used for spectral coefficient storage.
 *
 * Format: 1 sign bit, 4 exponent bits, 3 mantissa bits
 * Range: ±448.0, minimum subnormal: 2^-9 = 0.001953125
 * This matches NVIDIA's FP8 E4M3 format (Hopper/Ada).
 *
 * For pre-Hopper GPUs, we store as uint8_t and decode in the kernel.
 */
struct alignas(1) fp8_e4m3_t {
    uint8_t data;

    __host__ __device__ fp8_e4m3_t() : data(0) {}
    __host__ __device__ explicit fp8_e4m3_t(uint8_t raw) : data(raw) {}
};

/**
 * @brief Encode a float to FP8 E4M3 format.
 *
 * @param val Input float value.
 * @return fp8_e4m3_t Encoded FP8 value.
 *
 * Clamping: values outside ±448.0 are clamped. NaN maps to NaN encoding.
 * Rounding: round-to-nearest-even.
 */
__host__ __device__ inline fp8_e4m3_t float_to_fp8_e4m3(float val) {
    fp8_e4m3_t result;

    // Handle special cases
    uint32_t fbits;
    memcpy(&fbits, &val, sizeof(float));

    uint8_t sign = (fbits >> 31) & 1;
    uint32_t f_exp = (fbits >> 23) & 0xFF;
    uint32_t f_mant = fbits & 0x7FFFFF;

    // NaN check
    if (f_exp == 0xFF && f_mant != 0) {
        result.data = 0x7F; // FP8 NaN (all exponent + mantissa bits set, sign=0)
        return result;
    }

    float abs_val = fabsf(val);

    // Clamp to FP8 E4M3 max: 448.0
    if (abs_val > 448.0f) {
        abs_val = 448.0f;
    }

    // Zero
    if (abs_val == 0.0f) {
        result.data = sign << 7;
        return result;
    }

    // Compute exponent and mantissa for E4M3
    // Bias = 7 for E4M3
    int exp_val = (int)floorf(log2f(abs_val));
    exp_val = max(-6, min(8, exp_val)); // Clamp exponent range

    int biased_exp = exp_val + 7; // bias = 7

    // Handle subnormals (biased_exp <= 0)
    if (biased_exp <= 0) {
        // Subnormal: encode as biased_exp=0, mantissa represents value
        float subnorm_scale = powf(2.0f, -6.0f); // 2^(1 - bias) = 2^(-6)
        int mant = (int)roundf(abs_val / subnorm_scale * 8.0f); // 3-bit mantissa
        mant = min(mant, 7);
        result.data = (sign << 7) | (uint8_t)(mant & 0x07);
        return result;
    }

    // Normal number
    biased_exp = min(biased_exp, 15); // 4-bit exponent max
    float scale = powf(2.0f, (float)(biased_exp - 7));
    float normalized = abs_val / scale - 1.0f; // Remove implicit 1
    int mant = (int)roundf(normalized * 8.0f); // 3 mantissa bits → 8 levels
    mant = max(0, min(mant, 7));

    result.data = (sign << 7) | ((uint8_t)(biased_exp & 0x0F) << 3) | (uint8_t)(mant & 0x07);
    return result;
}

/**
 * @brief Decode an FP8 E4M3 value back to float.
 *
 * @param val FP8 encoded value.
 * @return float Decoded float value.
 */
__host__ __device__ inline float fp8_e4m3_to_float(fp8_e4m3_t val) {
    uint8_t sign = (val.data >> 7) & 1;
    uint8_t biased_exp = (val.data >> 3) & 0x0F;
    uint8_t mant = val.data & 0x07;

    float result;

    if (biased_exp == 0) {
        // Subnormal or zero
        if (mant == 0) {
            result = 0.0f;
        } else {
            // Subnormal: value = (-1)^sign * 2^(1-bias) * (mant/8)
            result = powf(2.0f, -6.0f) * ((float)mant / 8.0f);
        }
    } else if (biased_exp == 15 && mant == 7) {
        // NaN
        result = nanf("");
        return sign ? -result : result;
    } else {
        // Normal: value = (-1)^sign * 2^(exp-bias) * (1 + mant/8)
        result = powf(2.0f, (float)biased_exp - 7.0f) * (1.0f + (float)mant / 8.0f);
    }

    return sign ? -result : result;
}

// ============================================================================
// HALF ↔ FLOAT CONVERSION HELPERS
// ============================================================================

/**
 * @brief Convert half precision to float (device function).
 */
__device__ inline float half_to_float(half h) {
    return __half2float(h);
}

/**
 * @brief Convert float to half precision (device function).
 */
__device__ inline half float_to_half(float f) {
    return __float2half(f);
}

/**
 * @brief Convert BF16 to float (device function).
 */
__device__ inline float bf16_to_float(__nv_bfloat16 b) {
    return __bfloat162float(b);
}

/**
 * @brief Convert float to BF16 (device function).
 */
__device__ inline __nv_bfloat16 float_to_bf16(float f) {
    return __float2bfloat16(f);
}

// ============================================================================
// TIMING UTILITIES
// ============================================================================

/**
 * @brief GPU timer using CUDA events for precise kernel timing.
 */
struct CudaTimer {
    cudaEvent_t start_event, stop_event;
    float elapsed_ms;

    CudaTimer() : elapsed_ms(0.0f) {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_event, stream));
    }

    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    }

    float elapsed() const { return elapsed_ms; }
};

/**
 * @brief CPU wall-clock timer for host-side measurements.
 */
struct HostTimer {
    std::chrono::high_resolution_clock::time_point t_start;
    double elapsed_ms;

    HostTimer() : elapsed_ms(0.0) {}

    void start() {
        t_start = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        auto t_end = std::chrono::high_resolution_clock::now();
        elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }

    double elapsed() const { return elapsed_ms; }
};

// ============================================================================
// MEMORY UTILITIES
// ============================================================================

/**
 * @brief Allocate device memory with error checking.
 */
template <typename T>
T* cuda_malloc(size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

/**
 * @brief Free device memory with error checking.
 */
template <typename T>
void cuda_free(T* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

/**
 * @brief Copy host → device.
 */
template <typename T>
void cuda_copy_h2d(T* dst, const T* src, size_t count, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, count * sizeof(T),
                                cudaMemcpyHostToDevice, stream));
}

/**
 * @brief Copy device → host.
 */
template <typename T>
void cuda_copy_d2h(T* dst, const T* src, size_t count, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, count * sizeof(T),
                                cudaMemcpyDeviceToHost, stream));
}

/**
 * @brief Copy device → device.
 */
template <typename T>
void cuda_copy_d2d(T* dst, const T* src, size_t count, cudaStream_t stream = 0) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, count * sizeof(T),
                                cudaMemcpyDeviceToDevice, stream));
}

// ============================================================================
// MATH UTILITIES
// ============================================================================

/**
 * @brief Integer ceiling division.
 */
__host__ __device__ inline int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

/**
 * @brief Round up to next multiple of alignment.
 */
__host__ __device__ inline int round_up(int val, int alignment) {
    return cdiv(val, alignment) * alignment;
}

/**
 * @brief Warp-aligned thread count (rounds up to next warp boundary).
 */
__host__ __device__ inline int warp_align(int threads) {
    return round_up(threads, WARP_SIZE);
}

// ============================================================================
// WARP-LEVEL PRIMITIVES
// ============================================================================

/**
 * @brief Warp-level reduction (sum) using shuffle instructions.
 */
__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * @brief Warp-level reduction (max) using shuffle instructions.
 */
__device__ inline float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/**
 * @brief Block-level reduction (sum) using shared memory.
 * @param val Per-thread value to reduce.
 * @param shared Shared memory buffer of size [blockDim.x / WARP_SIZE].
 * @return Sum across all threads in the block (only valid in thread 0).
 */
__device__ inline float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;

    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

/**
 * @brief Block-level reduction (max) using shared memory.
 */
__device__ inline float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : -INFINITY;

    if (warp_id == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

// ============================================================================
// ONLINE SOFTMAX UTILITIES (for attention kernels)
// ============================================================================

/**
 * @brief Running max + sum-of-exp state for numerically stable online softmax.
 */
struct OnlineSoftmaxState {
    float max_val;
    float sum_exp;

    __device__ OnlineSoftmaxState() : max_val(-INFINITY), sum_exp(0.0f) {}

    __device__ void update(float val) {
        if (val > max_val) {
            sum_exp = sum_exp * expf(max_val - val) + expf(0.0f);
            max_val = val;
        } else {
            sum_exp += expf(val - max_val);
        }
    }

    __device__ void merge(const OnlineSoftmaxState& other) {
        if (other.max_val > max_val) {
            sum_exp = sum_exp * expf(max_val - other.max_val) + other.sum_exp;
            max_val = other.max_val;
        } else {
            sum_exp += other.sum_exp * expf(other.max_val - max_val);
        }
    }

    __device__ float normalize(float val) const {
        return expf(val - max_val) / sum_exp;
    }
};
