# Migration to candle-coreml

## Changes Made

### 1. Updated `Cargo.toml`
- Added `candle-coreml = "0.3"`
- Added `candle-core = { version = "0.9", features = ["metal"] }`
- Using version 0.9 instead of 0.4 (which had bf16/f16 compilation issues)

### 2. Created `src/models/coreml_model.rs`
- `CoreMLModelWrapper` struct to wrap Core ML models
- `load()` method to load `.mlmodelc` models with config
- `forward()` method for forward passes (input_ids → logits)
- `generate()` method placeholder for text generation

### 3. Updated `src/models/mod.rs`
- Added `coreml_model` module
- Exported `CoreMLModelWrapper`

## Next Steps

### 1. Test Compilation
```bash
cd rust_rlaif
cargo build --release
```

If compilation succeeds, the bf16/f16 issues are fixed in candle-core 0.9!

### 2. Convert Models to Core ML Format
Core ML models need to be in `.mlmodelc` format. You'll need to:
- Convert PyTorch/MLX models to Core ML format
- Save them with tokenizer in the same directory

### 3. Update StudentModel
Replace MLX generator with Core ML model:
```rust
// In student.rs
pub async fn with_coreml<P: AsRef<Path>>(
    model_path: P,
    vocab_size: usize,
    max_seq_len: usize,
) -> Result<Self> {
    let coreml_model = CoreMLModelWrapper::load(model_path, vocab_size, max_seq_len)?;
    // ...
}
```

### 4. Update Training Step
Replace MLX trainer with Core ML model in `training_step()`:
```rust
// Use Core ML for forward pass
let logits = coreml_model.forward(&input_ids_batch).await?;
```

## Benefits

- ✅ Native Rust implementation (no Python subprocess)
- ✅ Direct Core ML integration (optimized for Apple Silicon)
- ✅ Should work around candle-core 0.4 compilation issues
- ✅ Better performance than subprocess approach

## Potential Issues

1. **candle-core 0.9 may still have issues**: Need to test compilation
2. **Core ML model conversion**: Need tools to convert models
3. **Training support**: Core ML is primarily for inference; may need hybrid approach for training

## Testing

Once you can build:
1. Test model loading with a sample `.mlmodelc` file
2. Test forward pass with dummy input_ids
3. Integrate into training loop



