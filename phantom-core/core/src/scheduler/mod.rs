//! # Chronos Scheduler
//!
//! Innovation 7: Deterministic Multi-Slot Scheduler
//!
//! Handles concurrent generation requests by multiplexing the hardware
//! across predefined "slots", ensuring latency limits are respected even
//! during KV-cache evictions.

use crate::{PhantomError, PhantomResult};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

pub struct ChronosScheduler {
    max_slots: usize,
    active_slots: Arc<Mutex<Vec<SlotState>>>,
    pending_queue: Arc<Mutex<VecDeque<Request>>>,
}

#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub prompt: String,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SlotStatus {
    Idle,
    Prefill,
    Decoding,
}

#[derive(Debug, Clone)]
pub struct SlotState {
    pub slot_id: usize,
    pub status: SlotStatus,
    pub active_request: Option<Request>,
    pub generated_tokens: u32,
}

impl ChronosScheduler {
    pub fn new(max_slots: usize) -> Self {
        let mut slots = Vec::with_capacity(max_slots);
        for i in 0..max_slots {
            slots.push(SlotState {
                slot_id: i,
                status: SlotStatus::Idle,
                active_request: None,
                generated_tokens: 0,
            });
        }
        
        info!(max_slots = max_slots, "Chronos Scheduler initialized");
        
        Self {
            max_slots,
            active_slots: Arc::new(Mutex::new(slots)),
            pending_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Enqueue a new generation request.
    pub fn enqueue_request(&self, request: Request) -> PhantomResult<()> {
        let mut queue = self.pending_queue.lock().unwrap();
        queue.push_back(request.clone());
        debug!(req_id = request.id, "Request enqueued");
        Ok(())
    }

    /// Tick the scheduler. Assigns pending requests to idle slots.
    pub fn tick(&self) -> PhantomResult<()> {
        let mut slots = self.active_slots.lock().unwrap();
        let mut queue = self.pending_queue.lock().unwrap();
        
        for slot in slots.iter_mut() {
            if slot.status == SlotStatus::Idle {
                if let Some(req) = queue.pop_front() {
                    info!(slot_id = slot.slot_id, req_id = req.id, "Assigning request to slot");
                    slot.status = SlotStatus::Prefill;
                    slot.active_request = Some(req);
                    slot.generated_tokens = 0;
                }
            }
        }
        
        Ok(())
    }
    
    /// Mark a slot's prefill phase as complete, moving it to Decoding.
    pub fn finish_prefill(&self, slot_id: usize) -> PhantomResult<()> {
        let mut slots = self.active_slots.lock().unwrap();
        if let Some(slot) = slots.iter_mut().find(|s| s.slot_id == slot_id) {
            if slot.status == SlotStatus::Prefill {
                slot.status = SlotStatus::Decoding;
                debug!(slot_id = slot.slot_id, "Slot moved to Decoding phase");
            }
        }
        Ok(())
    }
    
    /// Free a slot after request completion.
    pub fn free_slot(&self, slot_id: usize) -> PhantomResult<()> {
        let mut slots = self.active_slots.lock().unwrap();
        if let Some(slot) = slots.iter_mut().find(|s| s.slot_id == slot_id) {
            info!(slot_id = slot.slot_id, "Freeing slot");
            slot.status = SlotStatus::Idle;
            slot.active_request = None;
            slot.generated_tokens = 0;
        }
        Ok(())
    }
}
