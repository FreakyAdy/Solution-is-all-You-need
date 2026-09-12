//! # PHANTOM CORE — Library Root
//!
//! Universal Hardware-Transcendent LLM Inference Engine
//!
//! This crate provides the core runtime for PHANTOM CORE, implementing
//! all seven innovations for running LLM models beyond hardware capacity.

pub mod engine;
pub mod ipc;
pub mod memory;
pub mod sampler;
pub mod scheduler;

use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Unified error type for all PHANTOM CORE operations.
#[derive(Error, Debug)]
pub enum PhantomError {
    /// CUDA runtime error
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// Memory allocation or management error
    #[error("Memory error: {0}")]
    MemoryError(String),

    /// NVMe I/O error
    #[error("NVMe I/O error: {0}")]
    NvmeError(String),

    /// Model loading error
    #[error("Model error: {0}")]
    ModelError(String),

    /// Calibration profile error
    #[error("Calibration error: {0}")]
    CalibrationError(String),

    /// IPC communication error
    #[error("IPC error: {0}")]
    IpcError(String),

    /// Scheduler error
    #[error("Scheduler error: {0}")]
    SchedulerError(String),

    /// Sampler error
    #[error("Sampler error: {0}")]
    SamplerError(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Configuration error
    #[error("Config error: {0}")]
    ConfigError(String),

    /// Generic I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// TOML error
    #[error("TOML error: {0}")]
    TomlError(String),

    /// Python interop error
    #[error("Python error: {0}")]
    PythonError(String),
}

/// Result type alias for PHANTOM CORE operations.
pub type PhantomResult<T> = Result<T, PhantomError>;

// ============================================================================
// Core Data Types
// ============================================================================

use serde::{Deserialize, Serialize};

/// Represents a tensor stored in device (GPU) memory.
#[derive(Debug, Clone)]
pub struct CudaTensor {
    /// Raw device pointer
    pub ptr: u64,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: TensorDtype,
    /// Size in bytes
    pub size_bytes: usize,
}

/// Tensor data types supported by PHANTOM CORE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorDtype {
    FP32,
    FP16,
    BF16,
    FP8E4M3,
    INT8,
    INT4,
}

impl TensorDtype {
    /// Size of one element in bytes.
    pub fn element_size(&self) -> usize {
        match self {
            TensorDtype::FP32 => 4,
            TensorDtype::FP16 | TensorDtype::BF16 => 2,
            TensorDtype::FP8E4M3 | TensorDtype::INT8 => 1,
            TensorDtype::INT4 => 1, // packed, but minimum addressable
        }
    }
}

/// Memory tier where a tensor or layer is resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum MemoryTier {
    /// GPU VRAM — fastest, most limited
    Vram,
    /// System RAM — medium speed, larger capacity
    Ram,
    /// NVMe SSD — slowest, largest capacity
    Nvme,
}

impl std::fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryTier::Vram => write!(f, "VRAM"),
            MemoryTier::Ram => write!(f, "RAM"),
            MemoryTier::Nvme => write!(f, "NVMe"),
        }
    }
}

/// Layer residency information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerResidency {
    pub layer_id: u32,
    pub layer_type: LayerType,
    pub tier: MemoryTier,
    pub size_bytes: usize,
    pub last_access_ms: u64,
    pub access_count: u64,
    pub pinned: bool,
}

/// Types of transformer layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    Attention,
    MLP,
    LayerNorm,
    Embedding,
    LMHead,
}

// ============================================================================
// Configuration
// ============================================================================

/// PHANTOM CORE engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Maximum VRAM to use (bytes). 0 = auto-detect.
    pub max_vram_bytes: u64,
    /// Maximum system RAM to use (bytes). 0 = auto-detect.
    pub max_ram_bytes: u64,
    /// Path to NVMe swap file.
    pub nvme_swap_path: String,
    /// Maximum NVMe space to use (bytes).
    pub max_nvme_bytes: u64,
    /// Number of layers to keep in VRAM (hot layers).
    pub hot_layer_count: u32,
    /// Number of layers to keep in RAM (warm layers).
    pub warm_layer_count: u32,
    /// Enable Wraith predictive prefetching.
    pub enable_wraith: bool,
    /// Enable Spectral Quantization.
    pub enable_spectral_quant: bool,
    /// Enable Neural Cache (KV compression).
    pub enable_neural_cache: bool,
    /// Enable Adaptive Compute Routing (sparsity).
    pub enable_sparse_routing: bool,
    /// Enable Resonance Sampling.
    pub enable_resonance_sampler: bool,
    /// Maximum concurrent model slots (Chronos scheduler).
    pub max_model_slots: u32,
    /// Log level: trace, debug, info, warn, error
    pub log_level: String,
    /// API server port.
    pub api_port: u16,
    /// Maximum concurrent generations.
    pub max_concurrent_generations: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_vram_bytes: 0,
            max_ram_bytes: 0,
            nvme_swap_path: String::from("~/.phantom/phantom_swap.bin"),
            max_nvme_bytes: 100 * 1024 * 1024 * 1024, // 100 GB
            hot_layer_count: 10,
            warm_layer_count: 30,
            enable_wraith: true,
            enable_spectral_quant: true,
            enable_neural_cache: true,
            enable_sparse_routing: true,
            enable_resonance_sampler: true,
            max_model_slots: 2,
            log_level: String::from("info"),
            api_port: 8080,
            max_concurrent_generations: 1,
        }
    }
}

/// Model architecture configuration (auto-detected or from profile).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name/identifier
    pub name: String,
    /// Total number of transformer layers
    pub num_layers: u32,
    /// Hidden dimension
    pub hidden_dim: u32,
    /// Number of attention heads (query)
    pub num_heads: u32,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: u32,
    /// FFN intermediate dimension
    pub ffn_dim: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Vocabulary size
    pub vocab_size: u32,
    /// Maximum sequence length
    pub max_seq_len: u32,
    /// RoPE base frequency
    pub rope_base: f32,
    /// Model architecture type
    pub arch_type: String,
    /// Total parameter count
    pub total_params: u64,
    /// Size per layer in bytes (approximate, FP16)
    pub layer_size_bytes: u64,
}

/// Calibration profile (loaded from .phantom file).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhantomProfile {
    /// Model hash (for verification)
    pub model_hash: String,
    /// Model config snapshot
    pub model_config: ModelConfig,
    /// Spectral K map: layer_id -> K value
    pub spectral_k_map: std::collections::HashMap<u32, u16>,
    /// Sparsity gate config: layer_id -> (enabled, threshold)
    pub gate_config: std::collections::HashMap<u32, (bool, f32)>,
    /// Average sparsity per layer
    pub avg_sparsity: std::collections::HashMap<u32, f32>,
    /// Paths to binary payloads
    pub kv_autoencoder_path: String,
    pub gate_weights_path: String,
    pub wraith_init_path: String,
    /// Hardware tier detected during calibration
    pub hardware_tier: String,
    /// Calibration timestamp
    pub calibrated_at: String,
}

/// Sampling parameters for text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub max_tokens: u32,
    pub repetition_penalty: f32,
    pub stop_sequences: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 2048,
            repetition_penalty: 1.1,
            stop_sequences: vec![],
        }
    }
}

/// Memory usage report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReport {
    pub vram_used_mb: f32,
    pub vram_total_mb: f32,
    pub ram_used_mb: f32,
    pub ram_total_mb: f32,
    pub nvme_used_mb: f32,
    pub nvme_total_mb: f32,
    pub hot_layers: Vec<u32>,
    pub warm_layers: Vec<u32>,
    pub cold_layers: Vec<u32>,
    pub wraith_accuracy_pct: f32,
    pub kv_compression_ratio: f32,
    pub active_sparsity_pct: f32,
    pub tok_per_sec: f32,
    pub thermal_state: String,
    pub throttle_active: bool,
}
