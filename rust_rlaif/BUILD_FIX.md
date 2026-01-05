# Build Fix Instructions

## The Problem

`candle-core` v0.4 has compilation errors due to trait bound issues with `bf16` and `f16` types. The errors occur because:
- `candle-core` uses both `rand` 0.8 and `rand` 0.9 (through `candle-metal-kernels`)
- Half-precision types don't implement required traits (`SampleBorrow`, `SampleUniform`, etc.)

## Solution Applied

1. **Removed candle dependencies** from `Cargo.toml` (commented out)
2. **Removed Cargo.lock** to force clean dependency resolution
3. **Modified code** to work without candle-core

## To Build Successfully

After removing Cargo.lock, run:

```bash
cd rust_rlaif
cargo clean          # Clean build cache
rm -f Cargo.lock     # Remove lock file (if it exists)
cargo build --release
```

If you still see `candle-core` being compiled, it means:
- The lock file wasn't removed, OR
- Another dependency is pulling it in

## Verification

Check that candle is not in the dependency tree:
```bash
cargo tree | grep -i candle
```

Should return nothing if candle is properly removed.

## Current Status

- ✅ Configuration loading works
- ✅ Teacher model (API) works
- ✅ Dataset loading works
- ✅ Reward computation works
- ❌ Student model (local) disabled (requires candle-core)
- ❌ Local generation disabled (requires candle-core)

## Alternative: Use Python Version

For actual training, use the fully functional Python version:
```bash
python scripts/training/train_rlaif.py --config config.yaml
```



