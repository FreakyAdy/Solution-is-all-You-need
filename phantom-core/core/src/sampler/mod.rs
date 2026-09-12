//! # Resonance Sampler
//!
//! Innovation 6: A specialized sampling strategy that penalizes repetitive
//! looping across long contexts, preventing the "hallucination loop" common
//! in heavily quantized or compressed models.

use crate::PhantomResult;
use std::collections::HashMap;
use tracing::debug;

pub struct ResonanceSampler {
    vocab_size: u32,
    token_history: Vec<u32>,
    /// Tracks distance to last occurrence of a token
    history_map: HashMap<u32, usize>,
    /// Base penalty applied to repeated tokens
    base_penalty: f32,
    /// Multiplier that increases as a token is repeated more often
    resonance_factor: f32,
    /// Max context to consider for repetition
    lookback_window: usize,
}

impl ResonanceSampler {
    pub fn new(vocab_size: u32) -> Self {
        Self {
            vocab_size,
            token_history: Vec::new(),
            history_map: HashMap::new(),
            base_penalty: 1.2,
            resonance_factor: 0.1,
            lookback_window: 1024,
        }
    }

    /// Apply resonance penalties to the logits.
    /// 
    /// Modifies the logits in-place.
    pub fn apply_penalties(&self, logits: &mut [f32]) -> PhantomResult<()> {
        let current_pos = self.token_history.len();
        
        // Count frequencies in the lookback window
        let mut freq_map = HashMap::new();
        let start_idx = current_pos.saturating_sub(self.lookback_window);
        
        for &token in &self.token_history[start_idx..] {
            *freq_map.entry(token).or_insert(0) += 1;
        }

        // Apply penalty: logit = logit / (base_penalty + resonance_factor * frequency)
        for (token, freq) in freq_map {
            if (token as usize) < logits.len() {
                let logit = logits[token as usize];
                
                // Only penalize positive logits (standard approach)
                if logit > 0.0 {
                    let penalty = self.base_penalty + (self.resonance_factor * freq as f32);
                    logits[token as usize] = logit / penalty;
                } else {
                    let penalty = self.base_penalty + (self.resonance_factor * freq as f32);
                    logits[token as usize] = logit * penalty;
                }
            }
        }
        
        Ok(())
    }

    /// Record a sampled token.
    pub fn record_token(&mut self, token: u32) {
        let pos = self.token_history.len();
        self.token_history.push(token);
        self.history_map.insert(token, pos);
    }
    
    /// Reset sampler state (e.g., for a new request).
    pub fn reset(&mut self) {
        self.token_history.clear();
        self.history_map.clear();
    }
}
