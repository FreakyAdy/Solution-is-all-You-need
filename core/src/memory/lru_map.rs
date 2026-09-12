//! # LRU Map — Persistent Hot/Cold Layer Map
//!
//! Tracks access frequency and recency for transformer layers across
//! all three memory tiers. The LRU state persists between runs, so
//! PHANTOM CORE learns which layers are frequently accessed and can
//! pre-warm them on subsequent launches.

use crate::{MemoryTier, PhantomError, PhantomResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

/// Access record for a single layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAccessRecord {
    /// Layer ID
    pub layer_id: u32,
    /// Total access count (lifetime)
    pub access_count: u64,
    /// Access count in current session
    pub session_access_count: u64,
    /// Exponential moving average of access frequency (accesses per second)
    pub ema_frequency: f64,
    /// Last access timestamp (ms since engine start)
    pub last_access_ms: u64,
    /// Current memory tier
    pub current_tier: MemoryTier,
    /// Predicted classification: hot, warm, or cold
    pub classification: LayerHeat,
}

/// Layer temperature classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerHeat {
    /// Frequently accessed — should be in VRAM
    Hot,
    /// Moderately accessed — should be in RAM
    Warm,
    /// Rarely accessed — can be on NVMe
    Cold,
}

/// Persistent LRU Map for layer access tracking.
pub struct LruMap {
    /// Access records per layer
    records: Arc<Mutex<HashMap<u32, LayerAccessRecord>>>,
    /// Path for persistence
    persist_path: PathBuf,
    /// Engine start time (for relative timestamps)
    start_time: std::time::Instant,
    /// EMA decay factor (higher = more weight on recent accesses)
    ema_alpha: f64,
    /// Hot threshold (accesses per second above this = hot)
    hot_threshold: f64,
    /// Warm threshold (above this = warm, below = cold)
    warm_threshold: f64,
}

impl LruMap {
    /// Create a new LRU map with persistence.
    ///
    /// # Arguments
    /// * `persist_path` - Path to save/load the LRU state
    /// * `hot_threshold` - EMA frequency threshold for "hot" classification
    /// * `warm_threshold` - EMA frequency threshold for "warm" classification
    pub fn new(persist_path: PathBuf, hot_threshold: f64, warm_threshold: f64) -> Self {
        let map = Self {
            records: Arc::new(Mutex::new(HashMap::new())),
            persist_path,
            start_time: std::time::Instant::now(),
            ema_alpha: 0.1,
            hot_threshold,
            warm_threshold,
        };

        // Try to load persisted state
        map.load_from_disk();
        map
    }

    /// Record an access to a layer.
    ///
    /// Updates access count, EMA frequency, last access time, and
    /// reclassifies the layer's temperature.
    pub fn record_access(&self, layer_id: u32) {
        let now_ms = self.start_time.elapsed().as_millis() as u64;

        let mut records = match self.records.lock() {
            Ok(r) => r,
            Err(_) => return,
        };

        let record = records.entry(layer_id).or_insert_with(|| LayerAccessRecord {
            layer_id,
            access_count: 0,
            session_access_count: 0,
            ema_frequency: 0.0,
            last_access_ms: now_ms,
            current_tier: MemoryTier::Nvme,
            classification: LayerHeat::Cold,
        });

        // Update access count
        record.access_count += 1;
        record.session_access_count += 1;

        // Update EMA frequency
        let elapsed_sec = (now_ms - record.last_access_ms).max(1) as f64 / 1000.0;
        let instant_freq = 1.0 / elapsed_sec;
        record.ema_frequency =
            self.ema_alpha * instant_freq + (1.0 - self.ema_alpha) * record.ema_frequency;

        record.last_access_ms = now_ms;

        // Reclassify
        record.classification = if record.ema_frequency >= self.hot_threshold {
            LayerHeat::Hot
        } else if record.ema_frequency >= self.warm_threshold {
            LayerHeat::Warm
        } else {
            LayerHeat::Cold
        };
    }

    /// Update a layer's current memory tier.
    pub fn update_tier(&self, layer_id: u32, tier: MemoryTier) {
        let mut records = match self.records.lock() {
            Ok(r) => r,
            Err(_) => return,
        };

        if let Some(record) = records.get_mut(&layer_id) {
            record.current_tier = tier;
        }
    }

    /// Get the classification of a layer.
    pub fn get_classification(&self, layer_id: u32) -> LayerHeat {
        self.records
            .lock()
            .ok()
            .and_then(|r| r.get(&layer_id).map(|rec| rec.classification))
            .unwrap_or(LayerHeat::Cold)
    }

    /// Get sorted layer IDs by temperature (hottest first).
    pub fn get_hot_layers(&self, count: usize) -> Vec<u32> {
        let records = match self.records.lock() {
            Ok(r) => r,
            Err(_) => return vec![],
        };

        let mut layers: Vec<(u32, f64)> = records
            .values()
            .map(|r| (r.layer_id, r.ema_frequency))
            .collect();

        layers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        layers.into_iter().take(count).map(|(id, _)| id).collect()
    }

    /// Get layers classified as cold.
    pub fn get_cold_layers(&self) -> Vec<u32> {
        let records = match self.records.lock() {
            Ok(r) => r,
            Err(_) => return vec![],
        };

        records
            .values()
            .filter(|r| r.classification == LayerHeat::Cold)
            .map(|r| r.layer_id)
            .collect()
    }

    /// Get a recommended tier placement for initial model loading.
    ///
    /// Returns three lists: (hot_layers, warm_layers, cold_layers)
    /// based on historical access patterns from previous sessions.
    pub fn get_initial_placement(
        &self,
        num_layers: u32,
        vram_slots: u32,
        ram_slots: u32,
    ) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let records = match self.records.lock() {
            Ok(r) => r,
            Err(_) => {
                // No history — use sequential placement
                let hot: Vec<u32> = (0..vram_slots.min(num_layers)).collect();
                let warm: Vec<u32> =
                    (vram_slots..vram_slots.saturating_add(ram_slots).min(num_layers)).collect();
                let cold: Vec<u32> =
                    (vram_slots.saturating_add(ram_slots)..num_layers).collect();
                return (hot, warm, cold);
            }
        };

        // Sort by EMA frequency
        let mut layers: Vec<(u32, f64)> = (0..num_layers)
            .map(|id| {
                let freq = records
                    .get(&id)
                    .map(|r| r.ema_frequency)
                    .unwrap_or(0.0);
                (id, freq)
            })
            .collect();

        layers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let hot: Vec<u32> = layers
            .iter()
            .take(vram_slots as usize)
            .map(|(id, _)| *id)
            .collect();
        let warm: Vec<u32> = layers
            .iter()
            .skip(vram_slots as usize)
            .take(ram_slots as usize)
            .map(|(id, _)| *id)
            .collect();
        let cold: Vec<u32> = layers
            .iter()
            .skip((vram_slots + ram_slots) as usize)
            .map(|(id, _)| *id)
            .collect();

        info!(
            hot_count = hot.len(),
            warm_count = warm.len(),
            cold_count = cold.len(),
            "Initial placement from historical access patterns"
        );

        (hot, warm, cold)
    }

    /// Save LRU state to disk.
    pub fn save_to_disk(&self) -> PhantomResult<()> {
        let records = self.records.lock().map_err(|e| {
            PhantomError::SerializationError(format!("Lock poisoned: {}", e))
        })?;

        let serialized = rmp_serde::to_vec(&*records).map_err(|e| {
            PhantomError::SerializationError(format!("MessagePack error: {}", e))
        })?;

        std::fs::write(&self.persist_path, &serialized).map_err(|e| {
            PhantomError::IoError(e)
        })?;

        debug!(
            path = %self.persist_path.display(),
            entries = records.len(),
            "LRU map saved to disk"
        );

        Ok(())
    }

    /// Load LRU state from disk.
    fn load_from_disk(&self) {
        match std::fs::read(&self.persist_path) {
            Ok(data) => {
                match rmp_serde::from_slice::<HashMap<u32, LayerAccessRecord>>(&data) {
                    Ok(loaded) => {
                        let mut records = self.records.lock().unwrap();
                        // Reset session counts but keep lifetime data
                        for (id, mut rec) in loaded {
                            rec.session_access_count = 0;
                            records.insert(id, rec);
                        }
                        info!(
                            entries = records.len(),
                            "LRU map loaded from disk (previous session data)"
                        );
                    }
                    Err(e) => {
                        debug!("Failed to parse LRU state: {}, starting fresh", e);
                    }
                }
            }
            Err(_) => {
                debug!("No LRU state file found, starting fresh");
            }
        }
    }

    /// Get summary statistics.
    pub fn stats(&self) -> (usize, usize, usize) {
        let records = match self.records.lock() {
            Ok(r) => r,
            Err(_) => return (0, 0, 0),
        };

        let hot = records.values().filter(|r| r.classification == LayerHeat::Hot).count();
        let warm = records.values().filter(|r| r.classification == LayerHeat::Warm).count();
        let cold = records.values().filter(|r| r.classification == LayerHeat::Cold).count();

        (hot, warm, cold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_access_tracking() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("lru.msgpack");
        let map = LruMap::new(path, 10.0, 1.0);

        // Access layer 0 many times rapidly
        for _ in 0..100 {
            map.record_access(0);
        }

        assert_eq!(map.get_classification(0), LayerHeat::Hot);
    }

    #[test]
    fn test_initial_placement() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("lru.msgpack");
        let map = LruMap::new(path, 10.0, 1.0);

        let (hot, warm, cold) = map.get_initial_placement(80, 10, 30);
        assert_eq!(hot.len(), 10);
        assert_eq!(warm.len(), 30);
        assert_eq!(cold.len(), 40);
    }
}
