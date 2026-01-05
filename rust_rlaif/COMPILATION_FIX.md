# Fixing candle-core Compilation Errors

## The Problem

`candle-core` v0.4 has trait bound errors:
```
error[E0277]: the trait bound `bf16: SampleBorrow<bf16>` is not satisfied
error[E0277]: the trait bound `bf16: SampleUniform` is not satisfied
```

This is caused by `candle-core` using both `rand` 0.8 and `rand` 0.9, and half-precision types not implementing required traits.

## Solution Applied

1. **Disabled candle-core dependency** in `Cargo.toml`
2. **Modified code** to work without candle-core:
   - Student model is a placeholder
   - Generation uses teacher model (API) instead
   - Device setup is disabled

## To Re-enable (When Fixed)

1. Uncomment candle-core in `Cargo.toml`:
   ```toml
   candle-core = { version = "0.4", features = ["metal"] }
   candle-nn = "0.4"
   candle-transformers = "0.4"
   ```

2. Restore device setup in `src/training/trainer.rs`

3. Implement full model loading in `src/models/student.rs`

4. Re-enable generation in `src/training/trainer.rs::generate_samples`

## Alternative: Use Python Version

The Python implementation is fully functional and recommended for actual training:
```bash
python scripts/training/train_rlaif.py --config config.yaml
```



