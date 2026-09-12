# ============================================================================
# PHANTOM CORE — CUDA Kernel Build System
# ============================================================================
#
# Builds all custom CUDA kernels for the PHANTOM CORE inference engine.
# Automatically detects GPU compute capability for optimal code generation.
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build . --parallel
#
# ============================================================================

cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(phantom_kernels LANGUAGES CXX CUDA)

# Require CUDA 12.x minimum
find_package(CUDAToolkit 12.0 REQUIRED)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect GPU architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Detect at configure time
    include(FindCUDA)
    # Default to common architectures: Ampere (86), Ada Lovelace (89), Hopper (90)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures to compile for")
    message(STATUS "[PHANTOM CORE] Auto-selected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -DDEBUG")

# Include common headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

# ============================================================================
# Spectral Quantization Kernels (Innovation 2)
# ============================================================================
add_library(phantom_spectral_quant STATIC
    spectral_quant/dct_compress.cu
    spectral_quant/idct_reconstruct.cu
    spectral_quant/fisher_calibrate.cu
)
target_include_directories(phantom_spectral_quant PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/spectral_quant
)
set_target_properties(phantom_spectral_quant PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Neural Cache Kernels (Innovation 3)
# ============================================================================
add_library(phantom_neural_cache STATIC
    neural_cache/kv_encode.cu
    neural_cache/kv_decode.cu
    neural_cache/ae_train.cu
)
target_include_directories(phantom_neural_cache PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/neural_cache
)
set_target_properties(phantom_neural_cache PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Sparse MoE Kernels (Innovation 5)
# ============================================================================
add_library(phantom_sparse_moe STATIC
    sparse_moe/gate_predict.cu
    sparse_moe/sparse_matmul.cu
    sparse_moe/gate_calibrate.cu
)
target_include_directories(phantom_sparse_moe PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/sparse_moe
)
set_target_properties(phantom_sparse_moe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Attention Kernels (FlashAttention-3 + GQA)
# ============================================================================
add_library(phantom_attention STATIC
    attention/flash_attn_v3.cu
    attention/gqa_kernel.cu
)
target_include_directories(phantom_attention PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/attention
)
set_target_properties(phantom_attention PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ============================================================================
# Unified PHANTOM CORE kernel library
# ============================================================================
add_library(phantom_kernels INTERFACE)
target_link_libraries(phantom_kernels INTERFACE
    phantom_spectral_quant
    phantom_neural_cache
    phantom_sparse_moe
    phantom_attention
    CUDA::cublas
    CUDA::curand
)

# ============================================================================
# Install rules
# ============================================================================
install(TARGETS
    phantom_spectral_quant
    phantom_neural_cache
    phantom_sparse_moe
    phantom_attention
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY common/ DESTINATION include/phantom/common FILES_MATCHING PATTERN "*.h")

message(STATUS "[PHANTOM CORE] Kernel build configured successfully")
message(STATUS "[PHANTOM CORE] CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "[PHANTOM CORE] Build type: ${CMAKE_BUILD_TYPE}")
