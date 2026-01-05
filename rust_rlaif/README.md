# RLAIF Trainer (Rust Implementation)

A Rust implementation of the RLAIF (Reinforcement Learning from AI Feedback) training system for code generation models.

## Features

- **Teacher-Student Training**: Uses API-based teacher models (OpenAI/Anthropic) to score student model outputs
- **Reward Computation**: Parallel reward computation with caching
- **Advantage Normalization**: Per-prompt baseline subtraction and whitening
- **KL Penalty**: Adaptive KL divergence penalty for policy optimization
- **Curriculum Learning**: Optional curriculum learning with difficulty bucketing
- **Multi-backend Support**: Metal (MPS), CUDA, and CPU support via Candle

## Prerequisites

**Rust and Cargo must be installed first!**

### Install Rust

Run the installation script:
```bash
./install_rust.sh
```

Or install manually:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

See [INSTALL.md](INSTALL.md) for detailed instructions.

## Building

```bash
cd rust_rlaif
cargo build --release
```

## ⚠️ Known Issue: candle-core Compilation Errors

**Current Status**: The Rust implementation has a compilation issue with `candle-core` v0.4 due to trait bound problems with `bf16`/`f16` types. 

**Workaround**: 
- The code has been modified to work in **API-only mode** (teacher model only)
- Local model generation is disabled until `candle-core` is fixed or updated
- **For actual training, use the Python version** (`scripts/training/train_rlaif.py`)

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for details.

**What Works**:
- ✅ Configuration loading
- ✅ Teacher model (API) integration  
- ✅ Dataset loading
- ✅ Reward computation
- ✅ Training loop structure

**What Doesn't Work**:
- ❌ Student model loading (blocked by candle-core)
- ❌ Local model generation (blocked by candle-core)
- ❌ Full training with local models

## Usage

```bash
# Basic training
cargo run --release -- --config config.yaml

# With custom model
cargo run --release -- --config config.yaml --model Qwen/Qwen2.5-Coder-3B-Instruct

# With custom data files
cargo run --release -- --config config.yaml --train-file ./data/train.jsonl --eval-file ./data/eval.jsonl
```

## Configuration

The configuration file (YAML) should match the structure defined in `src/config.rs`. Key parameters:

- `base_model`: HuggingFace model identifier or local path
- `teacher_provider`: "anthropic" or "openai"
- `teacher_model`: Model name (e.g., "claude-3-5-haiku-20241022")
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `kl_penalty`: KL divergence penalty coefficient
- `num_samples_per_prompt`: Number of samples to generate per prompt

## Architecture

### Core Components

1. **Config** (`src/config.rs`): Configuration management with YAML loading
2. **Models** (`src/models/`):
   - `teacher.rs`: API-based teacher model for generation and scoring
   - `student.rs`: Local model for code generation
   - `dataset.rs`: Dataset loading from JSONL files
3. **Training** (`src/training/`):
   - `trainer.rs`: Main training loop
   - `rewards.rs`: Reward computation
   - `advantages.rs`: Advantage normalization
4. **Utils** (`src/utils/`): Utility functions

## Differences from Python Version

- **Async/Await**: Uses Tokio for async operations (API calls, parallel processing)
- **Type Safety**: Strong typing throughout
- **Memory Safety**: Rust's ownership system prevents common bugs
- **Performance**: Compiled code with optimizations
- **Candle Framework**: Uses Candle instead of PyTorch for ML operations

## Status

This is a foundational implementation. Full feature parity with the Python version would require:

- Complete autoregressive generation with proper sampling
- KV cache implementation
- Full optimizer and scheduler support
- TensorBoard logging
- MLX backend support
- Comprehensive error handling and recovery
- Checkpoint saving/loading
- More sophisticated reward computation

## License

Same as the main project.

