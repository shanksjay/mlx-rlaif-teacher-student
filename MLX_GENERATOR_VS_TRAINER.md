# MLX Generator vs MLX Trainer: Key Differences

Both `MlxGenerator` and `MlxTrainer` use the same model path (`./mlx_model/q4`), but they differ in several important ways:

## Architecture Differences

### MlxGenerator (Inference/Generation)
- **Implementation**: Python subprocess (`scripts/utils/mlx_gen_worker.py`)
- **Communication**: JSON Lines over stdin/stdout
- **Process Isolation**: Runs in separate process to isolate Metal allocations from PyTorch MPS
- **Purpose**: Fast forward passes for text generation

### MlxTrainer (Training)
- **Implementation**: Direct PyO3 calls (embedded Python)
- **Communication**: Direct function calls via PyO3
- **Process**: Same process as Rust code
- **Purpose**: Forward + backward passes + optimizer updates for training

## What They Load from `mlx_lm.load(model_path)`

Both call `mlx_lm.load(model_path)`, which returns `(model, tokenizer)`:

### MlxGenerator
```python
model, tokenizer = load(args.model_path)
```
- **Uses both**: Model for generation, tokenizer for encoding/decoding prompts
- **Tokenization**: Handles tokenization internally using the MLX tokenizer

### MlxTrainer
```rust
let result = load_fn.call1(args)?;  // Returns (model, tokenizer)
let model = model_tuple.get_item(0)?;  // Only extracts model
```
- **Uses only**: Model (first element of tuple)
- **Tokenization**: Uses a separate tokenizer loaded in `StudentModel` (from `tokenizer.json`)

## Why Different Tokenizers?

1. **MlxGenerator**: Uses the tokenizer from `mlx_lm.load()` because it needs to:
   - Encode prompts before generation
   - Decode generated token IDs back to text
   - The MLX tokenizer is optimized for MLX models

2. **MlxTrainer**: Uses a separate tokenizer (loaded from `tokenizer.json`) because:
   - Tokenization happens in Rust before passing to MLX
   - The Rust tokenizer is used for consistency across the codebase
   - Tokenization is done in `trainer.rs` before calling `train_step()`

## Model Usage

### MlxGenerator
- **Forward pass only**: `model.generate()` or similar
- **No gradients**: Model is in inference mode
- **Optimized for speed**: Uses optimized generation algorithms

### MlxTrainer
- **Forward + backward**: Computes gradients via `mlx.core.value_and_grad()`
- **Optimizer updates**: Updates model parameters using Adam optimizer
- **Training mode**: Model can be modified during training

## Summary

| Aspect | MlxGenerator | MlxTrainer |
|--------|--------------|------------|
| **Architecture** | Python subprocess | PyO3 embedded Python |
| **Model Loading** | `model, tokenizer = load()` | `model = load()[0]` |
| **Tokenizer** | Uses MLX tokenizer | Uses separate Rust tokenizer |
| **Purpose** | Inference (generation) | Training (backprop + optimizer) |
| **Gradients** | No | Yes |
| **Process** | Separate process | Same process |

## Why Both Exist?

1. **Separation of Concerns**: Generation and training have different requirements
2. **Process Isolation**: Generator subprocess prevents Metal allocation conflicts
3. **Performance**: Each is optimized for its specific use case
4. **Flexibility**: Can use different tokenizers if needed

