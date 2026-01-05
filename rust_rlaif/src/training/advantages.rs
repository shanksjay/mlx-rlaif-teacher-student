use anyhow::Result;
use crate::config::RlaifConfig;
use crate::training::trainer::GeneratedSample;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, warn};

pub fn compute_advantages(
    rewards: &[f64],
    samples: &[GeneratedSample],
    config: &Arc<RlaifConfig>,
) -> Result<Vec<f64>> {
    // Validate rewards
    let rewards: Vec<f64> = rewards.iter()
        .map(|r| {
            if r.is_nan() || r.is_infinite() {
                warn!("NaN/Inf detected in reward, replacing with 0.5");
                0.5
            } else {
                r.max(0.0).min(1.0) // Clamp to [0, 1]
            }
        })
        .collect();
    
    if !config.use_advantage_normalization {
        return Ok(rewards);
    }

    // Group rewards by prompt
    let mut prompt_rewards: HashMap<String, Vec<f64>> = HashMap::new();
    for (sample, reward) in samples.iter().zip(rewards.iter()) {
        prompt_rewards
            .entry(sample.prompt.clone())
            .or_insert_with(Vec::new)
            .push(*reward);
    }

    // Compute per-prompt advantages (A_i = r_i - mean(r for same prompt))
    let mut advantages = Vec::new();
    for (sample, reward) in samples.iter().zip(rewards.iter()) {
        if let Some(prompt_rewards) = prompt_rewards.get(&sample.prompt) {
            let mean_reward = prompt_rewards.iter().sum::<f64>() / prompt_rewards.len() as f64;
            let adv = reward - mean_reward;
            advantages.push(adv);
            debug!("Prompt: '{}...', reward: {:.4}, mean: {:.4}, advantage: {:.4}", 
                   &sample.prompt.chars().take(50).collect::<String>(), 
                   reward, mean_reward, adv);
        } else {
            advantages.push(*reward);
        }
    }

    // Check if all advantages are zero
    let all_zero = advantages.iter().all(|a| a.abs() < 1e-8);
    if all_zero {
        warn!("All advantages are zero after per-prompt normalization. This may indicate all samples from the same prompt have identical rewards.");
        warn!("Raw rewards: {:?}", rewards);
        // Fall back to using raw rewards (centered) to provide a learning signal
        let mean_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;
        let centered: Vec<f64> = rewards.iter().map(|r| r - mean_reward).collect();
        return Ok(centered);
    }

    // Whitening (normalize to zero mean, unit variance)
    let mean_adv = advantages.iter().sum::<f64>() / advantages.len() as f64;
    let variance = advantages.iter()
        .map(|a| (a - mean_adv).powi(2))
        .sum::<f64>() / advantages.len() as f64;
    let std_adv = variance.sqrt();
    
    // Check for near-zero variance
    if std_adv < 1e-8 {
        warn!("Advantage variance is near-zero (std: {:.8}). All advantages are identical.", std_adv);
        warn!("Raw advantages: {:?}", advantages);
        // Return centered advantages without whitening
        let centered: Vec<f64> = advantages.iter().map(|a| a - mean_adv).collect();
        return Ok(centered);
    }

    let whitened: Vec<f64> = advantages.iter()
        .map(|a| (a - mean_adv) / std_adv)
        .collect();
    
    debug!("Advantages - mean: {:.4}, std: {:.4}, min: {:.4}, max: {:.4}", 
           mean_adv, std_adv,
           whitened.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
           whitened.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

    Ok(whitened)
}

