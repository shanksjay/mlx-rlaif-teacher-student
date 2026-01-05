# Core ML Implementation Plan

## Goal
Use `candle-coreml` to load and run Core ML models (.mlmodelc) for training, avoiding the `candle-core` compilation issues.

## Current Status

### Issue
- `candle-core` v0.4 has compilation errors with `bf16`/`f16` types
- Need to verify if `candle-coreml` depends on `candle-core`
- If it does, we'll need an alternative approach

### Steps to Verify

1. **Check candle-coreml dependencies**:
   ```bash
   cargo tree -p candle-coreml
   ```
   This will show if it depends on `candle-core`.

2. **If candle-coreml is independent**:
   - Add `candle-coreml = "0.3"` to `Cargo.toml`
   - Implement `CoreMLModelWrapper` to load `.mlmodelc` models
   - Use for forward passes and inference

3. **If candle-coreml depends on candle-core**:
   - Option A: Wait for candle-core fix (0.5+)
   - Option B: Use direct Core ML bindings (coreml-rs or similar)
   - Option C: Continue with MLX subprocess approach

## Implementation Plan

### Phase 1: Verification
- [ ] Check if `candle-coreml` crate exists on crates.io
- [ ] Verify dependencies (does it need candle-core?)
- [ ] Test compilation with candle-coreml

### Phase 2: Model Loading
- [ ] Implement `CoreMLModelWrapper::load()` to load `.mlmodelc` models
- [ ] Load tokenizer from model directory
- [ ] Test model loading

### Phase 3: Forward Pass
- [ ] Implement `forward()` method to convert input_ids to Core ML format
- [ ] Run inference through Core ML model
- [ ] Convert outputs back to logits format

### Phase 4: Training Integration
- [ ] Replace MLX trainer with Core ML model in `StudentModel`
- [ ] Update `training_step()` to use Core ML forward pass
- [ ] Implement backward pass (if Core ML supports gradients)

## Alternative: Direct Core ML Bindings

If `candle-coreml` doesn't work, we could use:
- `coreml-rs`: Direct Rust bindings to Core ML
- `coremltools`: Convert models to Core ML format, then use native bindings

## Notes

- Core ML models are typically compiled/optimized for inference
- Training with Core ML may require special setup
- Core ML is primarily designed for inference, not training
- May need to use Core ML for forward passes, but handle gradients differently



