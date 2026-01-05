# MLX Training Approach

## Overview

The Rust RLAIF trainer uses **MLX for both forward and backward passes**:

- **MLX for Forward Passes (Generation)**: Fast inference using MLX models via Python subprocess
- **MLX for Backward Passes (Training)**: Complete training step (forward + backward + optimizer) using MLX `value_and_grad` and optimizers
- **Native Rust for Rewards/Scoring**: All reward computation and scoring stays in Rust (no subprocess overhead)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RLAIF Training Loop                      │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                         │
        ▼                                         ▼
┌───────────────┐                        ┌───────────────┐
│   Generation  │                        │    Training   │
│  (Forward)    │                        │ (Forward +    │
│               │                        │  Backward +   │
│               │                        │  Optimizer)   │
└───────────────┘                        └───────────────┘
        │                                         │
        ▼                                         ▼
┌───────────────┐                        ┌───────────────┐
│  MLX Generator│                        │  MLX Trainer  │
│ (via Python   │                        │ (via Python   │
│  subprocess)  │                        │  subprocess)  │
│               │                        │               │
│ mlx_lm.generate│                        │ value_and_grad│
│               │                        │ + optim.Adam  │
└───────────────┘                        └───────────────┘
        │                                         │
        └───────────────────┬───────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   Rewards &   │
                    │   Scoring    │
                    │ (Native Rust)│
                    └───────────────┘
```

## Components

### 1. MLX Generator (`src/models/mlx_generator.rs`)
- **Purpose**: Fast forward passes for text generation
- **Implementation**: Python subprocess (`scripts/utils/mlx_gen_worker.py`)
- **Protocol**: JSON Lines over stdin/stdout
- **Speed**: 5-10x faster than PyTorch MPS on Apple Silicon
- **Usage**: Called during `generate_samples()` for fast inference

### 2. MLX Trainer (`src/models/mlx_trainer.rs`)
- **Purpose**: Complete training step (forward + backward + optimizer)
- **Implementation**: Python subprocess (`scripts/utils/mlx_train_worker.py`)
- **Protocol**: JSON Lines over stdin/stdout
- **Usage**: Called during `training_step()` for:
  - Forward pass → logits
  - Loss computation (policy + KL)
  - Backward pass via `mx.value_and_grad`
  - Optimizer update via `optim.Adam`

### 3. Native Rust Rewards (`src/training/rewards.rs`)
- **Purpose**: Reward computation and scoring
- **Implementation**: Pure Rust, no subprocess
- **Usage**: Called after generation to score code samples

## Configuration

### config.yaml

```yaml
hardware:
  mlx_model_path: ./mlx_model/q4  # MLX model for both generation and training
  use_mlx_for_generation: true
```

### Model Paths

- **MLX Model**: Path to MLX model directory (created by `mlx_lm.convert`)
  - Used for both generation (via `MlxGenerator`) and training (via `MlxTrainer`)

## Workflow

### Generation Phase (Forward Pass)
1. `generate_samples()` is called
2. For each prompt, `StudentModel::generate()` is called
3. If MLX generator is available:
   - Format prompt
   - Send to MLX worker subprocess
   - Receive generated text
   - Return code sample
4. If MLX not available, fall back to API (teacher model)

### Training Phase (Forward + Backward + Optimizer)
1. `training_step()` is called with samples and advantages
2. Tokenize samples to get `input_ids`
3. Call `StudentModel::train_step()` which uses MLX trainer:
   - Send `input_ids` and `advantages` to MLX worker
   - MLX worker performs:
     - Forward pass → `logits`
     - Compute `log_probs` via `nn.log_softmax`
     - Compute policy loss: `-(log_probs * advantages).mean()`
     - Compute KL divergence (if reference model available)
     - Compute total loss: `policy_loss + kl_penalty * kl_divergence`
     - Backward pass via `mx.value_and_grad(loss_fn)`
     - Gradient clipping
     - Optimizer update via `optim.Adam.update(model, grads)`
   - Receive loss and gradient norm
4. All computation happens in MLX Python subprocess for speed

### Rewards/Scoring Phase
1. `compute_rewards()` is called (native Rust)
2. For each generated sample:
   - Send to teacher model API for scoring
   - Receive reward score
   - Cache result
3. All computation stays in Rust (no subprocess)

## Benefits

1. **Speed**: MLX is 5-10x faster than PyTorch MPS for both generation and training
2. **Unified**: Same MLX model used for both forward and backward passes
3. **Efficiency**: Rewards/scoring in native Rust (no subprocess overhead)
4. **Optimized**: MLX's `value_and_grad` and optimizers are highly optimized for Apple Silicon

## Model Conversion

### MLX Model
```bash
# Convert HuggingFace model to MLX format
uv run mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-Coder-3B-Instruct \
  --mlx-path ./mlx_model/q4 \
  -q --q-bits 4
```

### MLX Model (for both generation and training)
```bash
# Convert HuggingFace model to MLX format
uv run mlx_lm.convert \
  --hf-path Qwen/Qwen2.5-Coder-3B-Instruct \
  --mlx-path ./mlx_model/q4 \
  -q --q-bits 4
```

## Current Status

✅ **Implemented**:
- MLX generator for fast forward passes (generation)
- MLX trainer for complete training step (forward + backward + optimizer)
- Native Rust rewards/scoring
- Gradient accumulation support
- Gradient clipping
- Adam optimizer integration

## Next Steps

1. **KL Divergence with Reference Model**:
   - Load reference model in MLX worker
   - Compute actual KL divergence between policy and reference
   - Add to loss computation

2. **Performance tuning**:
   - Optimize subprocess communication
   - Reduce memory allocations in MLX worker
   - Benchmark training throughput

3. **Advanced Features**:
   - LoRA/QLoRA support in MLX
   - Mixed precision training
   - Gradient checkpointing

