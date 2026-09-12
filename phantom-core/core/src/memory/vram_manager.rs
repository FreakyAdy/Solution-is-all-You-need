//! # VRAM Manager — GPU Memory Allocation Tracker
//!
//! Tracks VRAM allocation, residency, and provides an allocation/eviction
//! interface for the inference engine. Integrates with NVML for real-time
//! VRAM monitoring.

use crate::{CudaTensor, LayerResidency, LayerType, MemoryTier, PhantomError, PhantomResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

/// VRAM allocation entry.
#[derive(Debug, Clone)]
struct VramAllocation {
    /// Layer ID that owns this allocation
    layer_id: u32,
    /// Device pointer
    ptr: u64,
    /// Size in bytes
    size_bytes: usize,
    /// When this allocation was created (monotonic timestamp)
    allocated_at: std::time::Instant,
    /// Whether this allocation is pinned (not evictable)
    pinned: bool,
}

/// VRAM Manager — tracks GPU memory usage and layer residency.
///
/// The manager maintains a map of which layers are currently resident in VRAM,
/// how much memory each uses, and provides eviction suggestions when VRAM
/// pressure is high.
pub struct VramManager {
    /// Total VRAM capacity in bytes
    total_vram_bytes: u64,
    /// Maximum usable VRAM (leaves headroom for CUDA overhead)
    max_usable_bytes: u64,
    /// Current allocations
    allocations: Arc<Mutex<HashMap<u32, VramAllocation>>>,
    /// Current total allocated bytes
    allocated_bytes: Arc<Mutex<u64>>,
    /// VRAM headroom fraction (reserve this much for CUDA runtime)
    headroom_fraction: f32,
}

impl VramManager {
    /// Create a new VRAM manager.
    ///
    /// # Arguments
    /// * `total_vram_bytes` - Total GPU VRAM in bytes (0 = auto-detect)
    /// * `headroom_fraction` - Fraction of VRAM to reserve for CUDA overhead (default 0.05)
    ///
    /// # Returns
    /// A configured VramManager instance.
    pub fn new(total_vram_bytes: u64, headroom_fraction: f32) -> PhantomResult<Self> {
        let total = if total_vram_bytes == 0 {
            detect_vram_size()?
        } else {
            total_vram_bytes
        };

        let headroom = (total as f32 * headroom_fraction) as u64;
        let max_usable = total.saturating_sub(headroom);

        info!(
            total_mb = total / (1024 * 1024),
            usable_mb = max_usable / (1024 * 1024),
            headroom_mb = headroom / (1024 * 1024),
            "VRAM Manager initialized"
        );

        Ok(Self {
            total_vram_bytes: total,
            max_usable_bytes: max_usable,
            allocations: Arc::new(Mutex::new(HashMap::new())),
            allocated_bytes: Arc::new(Mutex::new(0)),
            headroom_fraction,
        })
    }

    /// Register a layer as resident in VRAM.
    ///
    /// # Arguments
    /// * `layer_id` - Unique layer identifier
    /// * `tensor` - CUDA tensor data for this layer
    pub fn register_layer(&self, layer_id: u32, tensor: &CudaTensor) -> PhantomResult<()> {
        let mut allocs = self.allocations.lock().map_err(|e| {
            PhantomError::MemoryError(format!("Lock poisoned: {}", e))
        })?;
        let mut total = self.allocated_bytes.lock().map_err(|e| {
            PhantomError::MemoryError(format!("Lock poisoned: {}", e))
        })?;

        let alloc = VramAllocation {
            layer_id,
            ptr: tensor.ptr,
            size_bytes: tensor.size_bytes,
            allocated_at: std::time::Instant::now(),
            pinned: false,
        };

        *total += tensor.size_bytes as u64;
        allocs.insert(layer_id, alloc);

        debug!(
            layer_id = layer_id,
            size_mb = tensor.size_bytes / (1024 * 1024),
            total_used_mb = *total / (1024 * 1024),
            "Layer registered in VRAM"
        );

        Ok(())
    }

    /// Unregister a layer from VRAM (after eviction).
    pub fn unregister_layer(&self, layer_id: u32) -> PhantomResult<Option<usize>> {
        let mut allocs = self.allocations.lock().map_err(|e| {
            PhantomError::MemoryError(format!("Lock poisoned: {}", e))
        })?;
        let mut total = self.allocated_bytes.lock().map_err(|e| {
            PhantomError::MemoryError(format!("Lock poisoned: {}", e))
        })?;

        if let Some(alloc) = allocs.remove(&layer_id) {
            *total = total.saturating_sub(alloc.size_bytes as u64);
            debug!(layer_id = layer_id, freed_mb = alloc.size_bytes / (1024 * 1024), "Layer evicted from VRAM");
            Ok(Some(alloc.size_bytes))
        } else {
            Ok(None)
        }
    }

    /// Pin a layer so it cannot be evicted.
    pub fn pin_layer(&self, layer_id: u32) -> PhantomResult<()> {
        let mut allocs = self.allocations.lock().map_err(|e| {
            PhantomError::MemoryError(format!("Lock poisoned: {}", e))
        })?;

        if let Some(alloc) = allocs.get_mut(&layer_id) {
            alloc.pinned = true;
            info!(layer_id = layer_id, "Layer pinned to VRAM");
        }
        Ok(())
    }

    /// Unpin a layer, making it eligible for eviction.
    pub fn unpin_layer(&self, layer_id: u32) -> PhantomResult<()> {
        let mut allocs = self.allocations.lock().map_err(|e| {
            PhantomError::MemoryError(format!("Lock poisoned: {}", e))
        })?;

        if let Some(alloc) = allocs.get_mut(&layer_id) {
            alloc.pinned = false;
            info!(layer_id = layer_id, "Layer unpinned from VRAM");
        }
        Ok(())
    }

    /// Check if a layer is currently in VRAM.
    pub fn is_resident(&self, layer_id: u32) -> bool {
        self.allocations
            .lock()
            .map(|allocs| allocs.contains_key(&layer_id))
            .unwrap_or(false)
    }

    /// Get available VRAM bytes.
    pub fn available_bytes(&self) -> u64 {
        let used = *self.allocated_bytes.lock().unwrap_or_else(|e| e.into_inner());
        self.max_usable_bytes.saturating_sub(used)
    }

    /// Check if there's enough VRAM for a new allocation.
    pub fn can_allocate(&self, size_bytes: usize) -> bool {
        self.available_bytes() >= size_bytes as u64
    }

    /// Suggest layers to evict to free `needed_bytes` of VRAM.
    ///
    /// Returns layer IDs in eviction priority order (oldest first, unpinned only).
    pub fn suggest_evictions(&self, needed_bytes: usize) -> Vec<u32> {
        let allocs = match self.allocations.lock() {
            Ok(a) => a,
            Err(_) => return vec![],
        };

        // Sort by allocation time (oldest first), excluding pinned layers
        let mut candidates: Vec<(u32, std::time::Instant, usize)> = allocs
            .values()
            .filter(|a| !a.pinned)
            .map(|a| (a.layer_id, a.allocated_at, a.size_bytes))
            .collect();

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

    /// Get a list of all resident layers with their residency info.
    pub fn get_residency_map(&self) -> Vec<LayerResidency> {
        let allocs = match self.allocations.lock() {
            Ok(a) => a,
            Err(_) => return vec![],
        };

        allocs
            .values()
            .map(|a| LayerResidency {
                layer_id: a.layer_id,
                layer_type: LayerType::Attention, // Simplified; real impl tracks this
                tier: MemoryTier::Vram,
                size_bytes: a.size_bytes,
                last_access_ms: a.allocated_at.elapsed().as_millis() as u64,
                access_count: 0,
                pinned: a.pinned,
            })
            .collect()
    }

    /// Get total used VRAM in MB.
    pub fn used_mb(&self) -> f32 {
        let used = *self.allocated_bytes.lock().unwrap_or_else(|e| e.into_inner());
        used as f32 / (1024.0 * 1024.0)
    }

    /// Get total VRAM in MB.
    pub fn total_mb(&self) -> f32 {
        self.total_vram_bytes as f32 / (1024.0 * 1024.0)
    }
}

/// Detect GPU VRAM size using NVML.
fn detect_vram_size() -> PhantomResult<u64> {
    match nvml_wrapper::Nvml::init() {
        Ok(nvml) => {
            let device = nvml.device_by_index(0).map_err(|e| {
                PhantomError::CudaError(format!("Failed to get GPU device: {}", e))
            })?;
            let mem_info = device.memory_info().map_err(|e| {
                PhantomError::CudaError(format!("Failed to get memory info: {}", e))
            })?;
            info!(
                vram_gb = mem_info.total as f64 / (1024.0 * 1024.0 * 1024.0),
                "Auto-detected VRAM size"
            );
            Ok(mem_info.total)
        }
        Err(e) => {
            warn!("NVML init failed: {}, defaulting to 6GB VRAM", e);
            Ok(6 * 1024 * 1024 * 1024) // Default 6GB
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_manager_basic() {
        let manager = VramManager::new(6 * 1024 * 1024 * 1024, 0.05).unwrap();
        assert!(manager.available_bytes() > 0);
        assert!(!manager.is_resident(0));
    }

    #[test]
    fn test_register_unregister() {
        let manager = VramManager::new(6 * 1024 * 1024 * 1024, 0.05).unwrap();
        let tensor = CudaTensor {
            ptr: 0x1000,
            shape: vec![4096, 4096],
            dtype: crate::TensorDtype::FP16,
            size_bytes: 4096 * 4096 * 2,
        };

        manager.register_layer(0, &tensor).unwrap();
        assert!(manager.is_resident(0));

        manager.unregister_layer(0).unwrap();
        assert!(!manager.is_resident(0));
    }

    #[test]
    fn test_eviction_suggestion() {
        let manager = VramManager::new(1024 * 1024, 0.0).unwrap(); // 1MB total
        let tensor = CudaTensor {
            ptr: 0x1000,
            shape: vec![256, 256],
            dtype: crate::TensorDtype::FP16,
            size_bytes: 256 * 1024, // 256KB each
        };

        for i in 0..4 {
            let mut t = tensor.clone();
            t.ptr = 0x1000 + i as u64 * 0x1000;
            manager.register_layer(i, &t).unwrap();
        }

        // Pin layer 0
        manager.pin_layer(0).unwrap();

        // Ask to free 512KB — should suggest non-pinned layers
        let evictions = manager.suggest_evictions(512 * 1024);
        assert!(!evictions.contains(&0)); // Layer 0 is pinned
        assert!(evictions.len() >= 2);
    }
}
