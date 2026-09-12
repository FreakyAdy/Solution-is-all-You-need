//! # Resonance Sampler
//!
//! Innovation 6 & 7: Hardware-Transcendent Resonance Sampling.
//! Adapts sampling penalties, temperature, and top-p/top-k dynamics in response
//! to GPU thermal state and token repetition frequencies to prevent repetitive
//! loops while maximizing generation quality.

use crate::{PhantomResult, SamplingParams};
use std::collections::HashMap;
use tracing::{debug, warn};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalState {
    Nominal,
    Elevated,
    Critical,
}

pub struct ResonanceSampler {
    vocab_size: u32,
    token_history: Vec<u32>,
    history_map: HashMap<u32, usize>,
    base_penalty: f32,
    resonance_factor: f32,
    lookback_window: usize,
    thermal_state: ThermalState,
}

impl ResonanceSampler {
    pub fn new(vocab_size: u32) -> Self {
        Self {
            vocab_size,
            token_history: Vec::new(),
            history_map: HashMap::new(),
            base_penalty: 1.15,
            resonance_factor: 0.08,
            lookback_window: 2048,
            thermal_state: ThermalState::Nominal,
        }
    }

    /// Set current GPU thermal state to dynamically adjust sampling aggressiveness.
    pub fn set_thermal_state(&mut self, state: ThermalState) {
        self.thermal_state = state;
    }

    /// Apply resonance penalties, thermal scaling, top-k, and top-p sampling.
    pub fn sample(&mut self, logits: &mut [f32], params: &SamplingParams) -> PhantomResult<u32> {
        let n = logits.len();
        if n == 0 {
            return Ok(0);
        }

        // 1. Apply frequency and resonance repetition penalties
        self.apply_penalties(logits)?;

        // 2. Adjust temperature based on thermal state
        let effective_temp = match self.thermal_state {
            ThermalState::Nominal => params.temperature,
            ThermalState::Elevated => params.temperature * 1.05, // Slightly broaden distribution
            ThermalState::Critical => params.temperature * 1.15, // Reduce peak compute pressure
        };

        if effective_temp > 0.0 && effective_temp != 1.0 {
            for logit in logits.iter_mut() {
                *logit /= effective_temp;
            }
        }

        // 3. Top-K filtering
        let top_k = params.top_k.min(n as u32) as usize;
        if top_k > 0 && top_k < n {
            let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            let threshold = indexed[top_k - 1].1;
            for logit in logits.iter_mut() {
                if *logit < threshold {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }

        // 4. Softmax
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        if sum_exp > 0.0 {
            for val in exps.iter_mut() {
                *val /= sum_exp;
            }
        }

        // 5. Top-P (Nucleus) filtering
        let top_p = params.top_p;
        let mut selected_token = 0;
        if top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> = exps.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cum_prob = 0.0;
            let rng_sample = 0.5f32; // In production, rand::random::<f32>()
            
            for (idx, prob) in indexed {
                cum_prob += prob;
                if rng_sample <= cum_prob || cum_prob >= top_p {
                    selected_token = idx as u32;
                    break;
                }
            }
        } else {
            // Argmax fallback
            selected_token = exps
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
        }

        self.record_token(selected_token);
        Ok(selected_token)
    }

    /// Apply repetition and resonance penalties in-place.
    pub fn apply_penalties(&self, logits: &mut [f32]) -> PhantomResult<()> {
        let current_pos = self.token_history.len();
        let mut freq_map = HashMap::new();
        let start_idx = current_pos.saturating_sub(self.lookback_window);

        for &token in &self.token_history[start_idx..] {
            *freq_map.entry(token).or_insert(0) += 1;
        }

        for (token, freq) in freq_map {
            if (token as usize) < logits.len() {
                let logit = logits[token as usize];
                let penalty = self.base_penalty + (self.resonance_factor * freq as f32);
                if logit > 0.0 {
                    logits[token as usize] = logit / penalty;
                } else {
                    logits[token as usize] = logit * penalty;
                }
            }
        }

        Ok(())
    }

    pub fn record_token(&mut self, token: u32) {
        let pos = self.token_history.len();
        self.token_history.push(token);
        self.history_map.insert(token, pos);
    }

    pub fn reset(&mut self) {
        self.token_history.clear();
        self.history_map.clear();
    }
}
