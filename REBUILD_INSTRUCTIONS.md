# Fixing PyO3 Python Detection Issue

## Problem
PyO3 is using the wrong Python interpreter (UV-managed Python instead of venv Python where MLX is installed).

## Root Cause
PyO3 detects Python at **build time**, not runtime. Even if `PYO3_PYTHON` is set at runtime, the binary was already linked against a different Python during compilation.

## Solution: Rebuild with PYO3_PYTHON Set

1. **Set PYO3_PYTHON to your venv Python:**
   ```bash
   export PYO3_PYTHON="$PWD/.venv/bin/python"
   ```

2. **Clean and rebuild the Rust binary:**
   ```bash
   cd rust_rlaif
   cargo clean
   cargo build --release
   ```

3. **Verify the build used the correct Python:**
   ```bash
   # Check that PYO3_PYTHON was used during build
   echo "PYO3_PYTHON=$PYO3_PYTHON"
   ```

4. **Run training:**
   ```bash
   cd ..
   ./run_training.sh
   ```

## Alternative: Use the Run Script (Auto-rebuild)

The `run_training.sh` script will:
- Automatically detect and set `PYO3_PYTHON`
- Verify MLX is installed
- Warn if a rebuild is needed

However, you still need to rebuild manually if the binary was built with a different Python.

## Verification

After rebuilding, check the logs. You should see:
- `PYO3_PYTHON` pointing to `.venv/bin/python`
- `Python prefix` pointing to `.venv`
- MLX imports succeeding

If you still see errors, check:
1. Is `PYO3_PYTHON` set correctly?
2. Does that Python have MLX installed? (`$PYO3_PYTHON -c "import mlx_lm"`)
3. Was the binary rebuilt after setting `PYO3_PYTHON`?

