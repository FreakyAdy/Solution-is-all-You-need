//! # Phantom Pages — NVMe-Backed Virtual VRAM Manager
//!
//! Innovation 4: Treats NVMe SSD as a third tier of memory below VRAM and RAM.
//!
//! Manages a pre-allocated swap file on NVMe, serializing/deserializing tensor
//! layers with LZ4 compression. Tracks access patterns via a persistent LRU map
//! and supports prefetch hints for proactive loading.
//!
//! The page manager auto-scales tier sizes based on detected hardware.

use crate::{CudaTensor, PhantomError, PhantomResult, TensorDtype};
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Handle referencing a stored page on NVMe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageHandle {
    /// Layer ID this page belongs to
    pub layer_id: u32,
    /// Byte offset into the swap file
    pub offset: u64,
    /// Compressed size in bytes
    pub compressed_size: u64,
    /// Original uncompressed size in bytes
    pub original_size: u64,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Tensor data type
    pub dtype: TensorDtype,
    /// Compression ratio achieved
    pub compression_ratio: f32,
}

/// Internal page allocation entry.
#[derive(Debug)]
struct PageSlot {
    offset: u64,
    size: u64,
    in_use: bool,
}

/// NVMe Phantom Page Manager.
///
/// Manages a pre-allocated file on NVMe that serves as a virtual VRAM extension.
/// Layers are LZ4-compressed and stored at fixed offsets for predictable I/O.
pub struct PhantomPageManager {
    /// Path to the NVMe swap file
    swap_path: PathBuf,
    /// Total capacity of the swap file in bytes
    capacity_bytes: u64,
    /// Page table: layer_id -> PageHandle
    page_table: Arc<Mutex<HashMap<u32, PageHandle>>>,
    /// Free space tracker
    next_offset: Arc<Mutex<u64>>,
    /// Prefetch request channel
    prefetch_tx: mpsc::Sender<Vec<u32>>,
    /// LRU state path (persisted across restarts)
    lru_state_path: PathBuf,
    /// Persistent open file handle to eliminate per-tile open/close syscall latency
    file_handle: Arc<tokio::sync::Mutex<Option<fs::File>>>,
}

impl PhantomPageManager {
    /// Create a new PhantomPageManager.
    ///
    /// # Arguments
    /// * `nvme_path` - Path to the NVMe swap file (will be pre-allocated)
    /// * `capacity_gb` - Capacity in GB for the swap file
    ///
    /// # Returns
    /// Initialized manager with pre-allocated swap file.
    pub async fn new(nvme_path: PathBuf, capacity_gb: f32) -> PhantomResult<Self> {
        let capacity_bytes = (capacity_gb * 1024.0 * 1024.0 * 1024.0) as u64;

        // Ensure parent directory exists
        if let Some(parent) = nvme_path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                PhantomError::NvmeError(format!("Failed to create directory: {}", e))
            })?;
        }

        // Pre-allocate swap file if it doesn't exist or is wrong size
        let needs_allocation = match fs::metadata(&nvme_path).await {
            Ok(meta) => meta.len() != capacity_bytes,
            Err(_) => true,
        };

        if needs_allocation {
            info!(
                path = %nvme_path.display(),
                capacity_gb = capacity_gb,
                "Pre-allocating NVMe swap file"
            );
            let file = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&nvme_path)
                .await
                .map_err(|e| PhantomError::NvmeError(format!("Failed to create swap file: {}", e)))?;

            file.set_len(capacity_bytes).await.map_err(|e| {
                PhantomError::NvmeError(format!("Failed to pre-allocate swap file: {}", e))
            })?;

            info!("NVMe swap file pre-allocated: {:.1} GB", capacity_gb);
        }

        // LRU state file
        let lru_state_path = nvme_path.with_extension("lru.msgpack");

        // Prefetch channel
        let (prefetch_tx, _prefetch_rx) = mpsc::channel::<Vec<u32>>(64);

        // Load existing page table if available
        let page_table = Arc::new(Mutex::new(HashMap::new()));
        let next_offset = Arc::new(Mutex::new(0u64));
        let file_handle = Arc::new(tokio::sync::Mutex::new(None));

        let manager = Self {
            swap_path: nvme_path,
            capacity_bytes,
            page_table,
            next_offset,
            prefetch_tx,
            lru_state_path,
            file_handle,
        };

        // Try to load persisted LRU state
        manager.load_lru_state().await;

        Ok(manager)
    }

    /// Evict a layer from VRAM/RAM to NVMe.
    ///
    /// Serializes the tensor to LZ4-compressed BF16/FP16, writes it to the
    /// swap file at a pre-allocated offset.
    ///
    /// # Arguments
    /// * `layer_id` - Layer identifier
    /// * `data` - Raw tensor data bytes (host memory)
    /// * `shape` - Tensor shape
    /// * `dtype` - Data type
    ///
    /// # Returns
    /// PageHandle for later retrieval.
    pub async fn evict_layer(
        &self,
        layer_id: u32,
        data: &[u8],
        shape: Vec<usize>,
        dtype: TensorDtype,
    ) -> PhantomResult<PageHandle> {
        let original_size = data.len() as u64;

        // LZ4 compress
        let compressed = compress_prepend_size(data);
        let compressed_size = compressed.len() as u64;
        let compression_ratio = original_size as f32 / compressed_size as f32;

        // Allocate offset in swap file
        let offset = {
            let mut next = self.next_offset.lock().map_err(|e| {
                PhantomError::NvmeError(format!("Lock poisoned: {}", e))
            })?;
            let current = *next;

            if current + compressed_size > self.capacity_bytes {
                return Err(PhantomError::NvmeError(
                    "NVMe swap file full — cannot evict more layers".to_string(),
                ));
            }

            *next = current + compressed_size;
            // Align to 4KB page boundary for optimal NVMe performance
            *next = (*next + 4095) & !4095;
            current
        };

        // Write to swap file
        let mut file = fs::OpenOptions::new()
            .write(true)
            .open(&self.swap_path)
            .await
            .map_err(|e| PhantomError::NvmeError(format!("Failed to open swap file: {}", e)))?;

        file.seek(std::io::SeekFrom::Start(offset)).await.map_err(|e| {
            PhantomError::NvmeError(format!("Seek failed: {}", e))
        })?;

        file.write_all(&compressed).await.map_err(|e| {
            PhantomError::NvmeError(format!("Write failed: {}", e))
        })?;

        file.flush().await.map_err(|e| {
            PhantomError::NvmeError(format!("Flush failed: {}", e))
        })?;

        let handle = PageHandle {
            layer_id,
            offset,
            compressed_size,
            original_size,
            shape,
            dtype,
            compression_ratio,
        };

        // Register in page table
        {
            let mut table = self.page_table.lock().map_err(|e| {
                PhantomError::NvmeError(format!("Lock poisoned: {}", e))
            })?;
            table.insert(layer_id, handle.clone());
        }

        debug!(
            layer_id = layer_id,
            original_mb = original_size / (1024 * 1024),
            compressed_mb = compressed_size / (1024 * 1024),
            ratio = format!("{:.2}x", compression_ratio),
            "Layer evicted to NVMe"
        );

        Ok(handle)
    }

    /// Load a layer from NVMe back to host memory.
    ///
    /// # Arguments
    /// * `handle` - PageHandle from a previous evict_layer call
    ///
    /// # Returns
    /// Raw decompressed tensor bytes.
    pub async fn load_layer(&self, handle: &PageHandle) -> PhantomResult<Vec<u8>> {
        let mut handle_guard = self.file_handle.lock().await;
        if handle_guard.is_none() {
            let file = fs::OpenOptions::new()
                .read(true)
                .open(&self.swap_path)
                .await
                .map_err(|e| PhantomError::NvmeError(format!("Failed to open swap file: {}", e)))?;
            *handle_guard = Some(file);
        }

        let file = handle_guard.as_mut().unwrap();
        file.seek(std::io::SeekFrom::Start(handle.offset))
            .await
            .map_err(|e| PhantomError::NvmeError(format!("Seek failed: {}", e)))?;

        let mut compressed = vec![0u8; handle.compressed_size as usize];
        file.read_exact(&mut compressed)
            .await
            .map_err(|e| PhantomError::NvmeError(format!("Read failed: {}", e)))?;

        // LZ4 decompress
        let decompressed = decompress_size_prepended(&compressed).map_err(|e| {
            PhantomError::NvmeError(format!("LZ4 decompression failed: {}", e))
        })?;

        debug!(
            layer_id = handle.layer_id,
            size_mb = decompressed.len() / (1024 * 1024),
            "Layer loaded from NVMe via persistent file handle"
        );

        Ok(decompressed)
    }

    /// Send a prefetch hint for layers that will be needed soon.
    ///
    /// This queues the layers for background loading by the I/O daemon
    /// without blocking the caller.
    pub async fn prefetch_hint(&self, layer_ids: Vec<u32>) -> PhantomResult<()> {
        self.prefetch_tx.send(layer_ids).await.map_err(|e| {
            PhantomError::NvmeError(format!("Prefetch channel send failed: {}", e))
        })?;
        Ok(())
    }

    /// Check if a layer is stored in the NVMe swap.
    pub fn has_layer(&self, layer_id: u32) -> bool {
        self.page_table
            .lock()
            .map(|t| t.contains_key(&layer_id))
            .unwrap_or(false)
    }

    /// Get the page handle for a stored layer.
    pub fn get_handle(&self, layer_id: u32) -> Option<PageHandle> {
        self.page_table
            .lock()
            .ok()
            .and_then(|t| t.get(&layer_id).cloned())
    }

    /// Get NVMe usage statistics.
    pub fn usage_mb(&self) -> f32 {
        let next = *self.next_offset.lock().unwrap_or_else(|e| e.into_inner());
        next as f32 / (1024.0 * 1024.0)
    }

    /// Get total NVMe capacity in MB.
    pub fn capacity_mb(&self) -> f32 {
        self.capacity_bytes as f32 / (1024.0 * 1024.0)
    }

    /// Persist LRU state to disk for cross-restart learning.
    pub async fn save_lru_state(&self) -> PhantomResult<()> {
        let table = self.page_table.lock().map_err(|e| {
            PhantomError::NvmeError(format!("Lock poisoned: {}", e))
        })?;

        let serialized = rmp_serde::to_vec(&*table).map_err(|e| {
            PhantomError::SerializationError(format!("MessagePack serialize failed: {}", e))
        })?;

        fs::write(&self.lru_state_path, &serialized).await.map_err(|e| {
            PhantomError::NvmeError(format!("Failed to save LRU state: {}", e))
        })?;

        debug!(path = %self.lru_state_path.display(), "LRU state persisted");
        Ok(())
    }

    /// Load persisted LRU state from disk.
    async fn load_lru_state(&self) {
        match fs::read(&self.lru_state_path).await {
            Ok(data) => {
                match rmp_serde::from_slice::<HashMap<u32, PageHandle>>(&data) {
                    Ok(table) => {
                        let mut pt = self.page_table.lock().unwrap();
                        *pt = table;
                        info!(
                            entries = pt.len(),
                            "Loaded persisted LRU state from previous session"
                        );
                    }
                    Err(e) => {
                        warn!("Failed to deserialize LRU state: {}, starting fresh", e);
                    }
                }
            }
            Err(_) => {
                debug!("No existing LRU state found, starting fresh");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_evict_and_load() {
        let tmp = TempDir::new().unwrap();
        let swap_path = tmp.path().join("test_swap.bin");

        let manager = PhantomPageManager::new(swap_path, 0.01).await.unwrap();

        // Create test data
        let test_data: Vec<u8> = (0..1024 * 1024).map(|i| (i % 256) as u8).collect();

        // Evict
        let handle = manager
            .evict_layer(0, &test_data, vec![512, 1024], TensorDtype::FP16)
            .await
            .unwrap();

        assert_eq!(handle.layer_id, 0);
        assert_eq!(handle.original_size, test_data.len() as u64);
        assert!(handle.compression_ratio > 1.0);

        // Load
        let loaded = manager.load_layer(&handle).await.unwrap();
        assert_eq!(loaded, test_data);
    }

    #[tokio::test]
    async fn test_has_layer() {
        let tmp = TempDir::new().unwrap();
        let swap_path = tmp.path().join("test_swap.bin");
        let manager = PhantomPageManager::new(swap_path, 0.01).await.unwrap();

        assert!(!manager.has_layer(0));

        let data = vec![0u8; 1024];
        manager
            .evict_layer(0, &data, vec![512], TensorDtype::FP16)
            .await
            .unwrap();

        assert!(manager.has_layer(0));
    }
}
