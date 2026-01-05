# Known Issues

## candle-core Compilation Errors

**Issue**: `candle-core` v0.4 has trait bound errors with `bf16` and `f16` types:
```
error[E0277]: the trait bound `bf16: SampleBorrow<bf16>` is not satisfied
error[E0277]: the trait bound `bf16: SampleUniform` is not satisfied
```

**Root Cause**: 
- `candle-core` 0.4 uses both `rand` 0.8 and `rand` 0.9 (through `candle-metal-kernels`)
- Half-precision types (`bf16`, `f16`) don't implement required traits for random number generation in these versions

**Workarounds**:

1. **Use API-only mode** (current workaround):
   - Student model generation is disabled
   - Only teacher model (API) functionality works
   - Use Python version for full training

2. **Wait for candle-core 0.5+**:
   - This issue may be fixed in future versions
   - Check: https://github.com/huggingface/candle/issues

3. **Use Python version**:
   - The Python implementation (`scripts/training/train_rlaif.py`) is fully functional
   - Recommended for actual training until Rust version is fixed

**Status**: Blocked on upstream fix in candle-core

## Current Functionality

✅ **Working**:
- Configuration loading
- Teacher model (API) integration
- Dataset loading
- Reward computation
- Training loop structure

❌ **Not Working**:
- Student model loading (requires candle-core)
- Local model generation (requires candle-core)
- Full training with local models

## Future Work

Once candle-core is fixed or updated:
1. Re-enable `candle-core` dependency
2. Implement full student model loading
3. Complete autoregressive generation
4. Add optimizer and scheduler support



