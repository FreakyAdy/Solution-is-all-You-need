//! # CPU Offload Manager — System RAM Layer Staging
//!
//! Manages transformer layers resident in system RAM as a warm tier
//! between VRAM (hot) and NVMe (cold). Layers in RAM can be loaded
//! to VRAM much faster than from NVMe.

use crate::{LayerResidency, LayerType, MemoryTier, PhantomError, PhantomResult, TensorDtype};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

/// A layer stored in CPU (system RAM) memory.
#[derive(Debug)]
struct CpuLayerEntry {
    layer_id: u32,
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: TensorDtype,
    stored_at: std::time::Instant,
    access_count: u64,
}

/// CPU Offload Manager — manages layers in system RAM.
pub struct CpuOffloadManager {
    /// Maximum RAM budget in bytes
    max_ram_bytes: u64,
    /// Current used bytes
    used_bytes: Arc<Mutex<u64>>,
    /// Layer storage
    layers: Arc<Mutex<HashMap<u32, CpuLayerEntry>>>,
}

impl CpuOffloadManager {
    /// Create a new CPU offload manager.
    ///
    /// # Arguments
    /// * `max_ram_bytes` - Maximum system RAM to use for layer staging (0 = auto)
    pub fn new(max_ram_bytes: u64) -> PhantomResult<Self> {
        let max_ram = if max_ram_bytes == 0 {
            // Use 50% of available system RAM by default
            let sys = sysinfo::System::new_with_specifics(
                sysinfo::RefreshKind::new().with_memory(sysinfo::MemoryRefreshKind::new().with_ram()),
            );
            let available = sys.available_memory();
            available / 2
        } else {
            max_ram_bytes
        };

        info!(
            max_ram_gb = max_ram as f64 / (1024.0 * 1024.0 * 1024.0),
            "CPU offload manager initialized"
        );

        Ok(Self {
            max_ram_bytes: max_ram,
            used_bytes: Arc::new(Mutex::new(0)),
            layers: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Store a layer in system RAM.
    pub fn store_layer(
        &self,
        layer_id: u32,
        data: Vec<u8>,
        shape: Vec<usize>,
        dtype: TensorDtype,
    ) -> PhantomResult<()> {
        let size = data.len() as u64;

        let mut used = self.used_bytes.lock().map_err(|e| {
            PhantomError::MemoryError(format!("Lock poisoned: {}", e))
        })?;

        if *used + size > self.max_ram_bytes {
            return Err(PhantomError::MemoryError(
                "CPU offload: insufficient RAM budget".to_string(),
            ));
        }

        let entry = CpuLayerEntry {
            layer_id,
            data,
            shape,
            dtype,
            stored_at: std::time::Instant::now(),
            access_count: 0,
        };

        let mut layers = self.layers.lock().map_err(|e| {
            PhantomError::MemoryError(format!("Lock poisoned: {}", e))
        })?;

        // Remove old version if exists
        if let Some(old) = layers.remove(&layer_id) {
            *used -= old.data.len() as u64;
        }

        *used += size;
        layers.insert(layer_id, entry);

        debug!(
            layer_id = layer_id,
            size_mb = size / (1024 * 1024),
            "Layer stored in CPU RAM"
        );

        Ok(())
    }

    /// Retrieve a layer from system RAM.
    pub fn load_layer(&self, layer_id: u32) -> PhantomResult<(Vec<u8>, Vec<usize>, TensorDtype)> {
        let mut layers = self.layers.lock().map_err(|e| {
            PhantomError::MemoryError(format!("Lock poisoned: {}", e))
        })?;

        let entry = layers.get_mut(&layer_id).ok_or_else(|| {
            PhantomError::MemoryError(format!("Layer {} not in CPU RAM", layer_id))
        })?;

        entry.access_count += 1;
        Ok((entry.data.clone(), entry.shape.clone(), entry.dtype))
    }

    /// Remove a layer from system RAM.
    pub fn remove_layer(&self, layer_id: u32) -> PhantomResult<Option<Vec<u8>>> {
        let mut layers = self.layers.lock().map_err(|e| {
            PhantomError::MemoryError(format!("Lock poisoned: {}", e))
        })?;
        let mut used = self.used_bytes.lock().map_err(|e| {
            PhantomError::MemoryError(format!("Lock poisoned: {}", e))
        })?;

        if let Some(entry) = layers.remove(&layer_id) {
            *used -= entry.data.len() as u64;
            debug!(layer_id = layer_id, "Layer removed from CPU RAM");
            Ok(Some(entry.data))
        } else {
            Ok(None)
        }
    }

    /// Check if a layer is in CPU RAM.
    pub fn has_layer(&self, layer_id: u32) -> bool {
        self.layers
            .lock()
            .map(|l| l.contains_key(&layer_id))
            .unwrap_or(false)
    }

    /// Get CPU RAM usage in MB.
    pub fn used_mb(&self) -> f32 {
        let used = *self.used_bytes.lock().unwrap_or_else(|e| e.into_inner());
        used as f32 / (1024.0 * 1024.0)
    }

    /// Get total RAM budget in MB.
    pub fn total_mb(&self) -> f32 {
        self.max_ram_bytes as f32 / (1024.0 * 1024.0)
    }

    /// Get residency info for all layers in RAM.
    pub fn get_residency_map(&self) -> Vec<LayerResidency> {
        let layers = match self.layers.lock() {
            Ok(l) => l,
            Err(_) => return vec![],
        };

        layers
            .values()
            .map(|e| LayerResidency {
                layer_id: e.layer_id,
                layer_type: LayerType::Attention,
                tier: MemoryTier::Ram,
                size_bytes: e.data.len(),
                last_access_ms: e.stored_at.elapsed().as_millis() as u64,
                access_count: e.access_count,
                pinned: false,
            })
            .collect()
    }

    /// Suggest layers to evict from RAM (oldest, least accessed first).
    pub fn suggest_evictions(&self, needed_bytes: usize) -> Vec<u32> {
        let layers = match self.layers.lock() {
            Ok(l) => l,
            Err(_) => return vec![],
        };

        let mut candidates: Vec<(u32, u64, usize)> = layers
            .values()
            .map(|e| (e.layer_id, e.access_count, e.data.len()))
            .collect();

        // Sort by access count (least accessed first)
        candidates.sort_by_key(|c| c.1);

        let mut freed = 0usize;
        let mut evict_list = Vec::new();

        for (layer_id, _, size) in candidates {
            if freed >= needed_bytes {
                break;
            }
            evict_list.push(layer_id);
            freed += size;
        }

        evict_list
    }
}
