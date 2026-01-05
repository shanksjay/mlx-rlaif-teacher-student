# Training Step Implementation Guide

## Why Backprop is Currently Simulated

The Rust RLAIF trainer currently uses **simulated backpropagation** because:

1. **candle-core Compilation Issues**: The `candle-core` dependency (required for model loading) has compilation errors with `bf16`/`f16` types (see `KNOWN_ISSUES.md`)

2. **No Model Loading**: Without `candle-core`, we cannot:
   - Load model weights
   - Run forward passes
   - Compute logits/log probabilities
   - Perform actual backpropagation

3. **Current Workaround**: The implementation uses:
   - MLX for generation (via Python subprocess)
   - API (teacher model) for scoring
   - Simulated backprop timing for metrics

## What Real Implementation Requires

### 1. Model Infrastructure

```rust
// Need to implement:
struct StudentModel {
    model: Qwen2Model,           // Actual model weights
    ref_model: Qwen2Model,        // Frozen reference model
    tokenizer: Tokenizer,
    device: Device,
}

impl StudentModel {
    fn forward(&self, input_ids: &Tensor) -> Tensor {
        // Forward pass returning logits [B, T, V]
    }
    
    fn compute_log_probs(&self, logits: &Tensor, input_ids: &Tensor) -> Tensor {
        // Compute log probabilities for selected tokens
    }
}
```

### 2. Training Step Components

#### A. Forward Pass
```rust
// Tokenize samples
let input_ids = tokenizer.encode_batch(samples)?;  // [B, T]

// Policy model forward
let logits = student_model.forward(&input_ids)?;    // [B, T, V]
let log_probs = log_softmax(&logits);              // [B, T, V]
let selected_log_probs = select_log_probs(&log_probs, &input_ids);  // [B, T]

// Reference model forward (for KL divergence)
let ref_logits = ref_model.forward(&input_ids)?;    // [B, T, V]
let ref_log_probs = log_softmax(&ref_logits);      // [B, T, V]
let ref_selected_log_probs = select_log_probs(&ref_log_probs, &input_ids);
```

#### B. Policy Loss
```rust
// Policy loss: -log_prob * advantage
let reward_signal = advantages.to_tensor();  // [B]
let policy_loss_raw = -(selected_log_probs * reward_signal * mask).mean();
let policy_loss = config.reward_weight * policy_loss_raw;
```

#### C. KL Divergence
```rust
// KL divergence: KL(policy || reference)
let kl_div = compute_tokenwise_kl(
    &logits,      // Policy logits
    &ref_logits,  // Reference logits
    &mask         // Attention mask
);
let kl_penalty = config.kl_penalty * kl_div.mean();
```

#### D. Total Loss
```rust
let total_loss = policy_loss + kl_penalty;
let scaled_loss = total_loss / config.gradient_accumulation_steps;
```

#### E. Backpropagation
```rust
// Backward pass
scaled_loss.backward()?;  // Accumulate gradients

// Gradient clipping
clip_grad_norm(&model.parameters(), config.max_grad_norm)?;

// Optimizer step (when accumulation complete)
if accumulation_step == config.gradient_accumulation_steps {
    optimizer.step()?;
    optimizer.zero_grad()?;
    scheduler.step()?;
}
```

### 3. Required Dependencies

```toml
# Once candle-core is fixed, uncomment:
candle-core = { version = "0.4", features = ["metal"] }
candle-nn = "0.4"
candle-transformers = "0.4"  # For Qwen2 model support
```

### 4. Optimizer & Scheduler

```rust
use candle_nn::optim::{AdamW, Optimizer};
use candle_nn::scheduler::CosineAnnealingLR;

let optimizer = AdamW::new(
    &model.trainable_params(),
    config.learning_rate,
    config.weight_decay,
)?;

let scheduler = CosineAnnealingLR::new(
    optimizer.learning_rate(),
    total_steps,
    config.warmup_steps,
)?;
```

## Current Status

✅ **Implemented**:
- Training loop structure
- Sample generation (MLX/API)
- Reward computation
- Advantage calculation
- Metrics tracking
- Code saving

❌ **Blocked** (requires candle-core fix):
- Model loading
- Forward pass
- Log probability computation
- KL divergence computation
- Actual backpropagation
- Optimizer updates

## Migration Path

Once `candle-core` is fixed:

1. **Uncomment dependencies** in `Cargo.toml`
2. **Implement model loading** in `StudentModel::load()`
3. **Add forward pass** method to `StudentModel`
4. **Implement training_step** with real forward/backward
5. **Add optimizer** and scheduler setup
6. **Remove simulation** code

## Alternative: Use MLX for Training

Since MLX works for generation, we could potentially:
- Use MLX for forward passes (via Python subprocess)
- Compute gradients in Rust
- Update weights via MLX API

This would be a hybrid approach but would work around candle-core issues.



