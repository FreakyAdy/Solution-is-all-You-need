# GGUF Support Matrix in PHANTOM

PHANTOM includes a native, zero-dependency GGUF loader and SIMD dequantizer implemented in pure PyTorch and NumPy.

## Supported Quantization Types

| GGUF Type | Block Size | Dequant Throughput | Precision Target | Status |
|---|---|---|---|---|
| `Q4_0` | 32 | ≥ 3.5 GB/s | max abs err < 1e-4 | ✅ Production |
| `Q4_1` | 32 | ≥ 3.2 GB/s | max abs err < 1e-4 | ✅ Production |
| `Q4_K_M` | 256 | ≥ 2.8 GB/s | max abs err < 1e-4 | ✅ Production (Primary) |
| `Q4_K_S` | 256 | ≥ 2.8 GB/s | max abs err < 1e-4 | ✅ Production |
| `Q5_0` | 32 | ≥ 3.0 GB/s | max abs err < 1e-4 | ✅ Production |
| `Q5_1` | 32 | ≥ 3.0 GB/s | max abs err < 1e-4 | ✅ Production |
| `Q5_K_M` | 256 | ≥ 2.5 GB/s | max abs err < 1e-4 | ✅ Production |
| `Q6_K` | 256 | ≥ 2.5 GB/s | max abs err < 1e-4 | ✅ Production |
| `Q8_0` | 32 | ≥ 4.2 GB/s | max abs err < 1e-4 | ✅ Production |
| `Q8_1` | 32 | ≥ 4.0 GB/s | max abs err < 1e-4 | ✅ Production |
| `F16` | 1 | ≥ 6.0 GB/s | Exact | ✅ Production |
| `BF16` | 1 | ≥ 6.0 GB/s | Exact | ✅ Production |
| `F32` | 1 | ≥ 8.0 GB/s | Exact | ✅ Production |

## Architecture Auto-Detection

The loader parses GGUF metadata keys (`general.architecture`, `*.block_count`, etc.) to automatically configure the layer maps for:
- **LLaMA** (1, 2, 3, 3.1, 3.2, 3.3)
- **Mistral** (0.1, 0.2, 0.3)
- **Mixtral** (8x7B, 8x22B MoE)
- **Gemma** (1, 2)
- **Qwen** (1.5, 2, 2.5)
- **Phi** (2, 3, 3.5)
- **DeepSeek** (V2, V3 with MLA attention detection)
- **Falcon**
