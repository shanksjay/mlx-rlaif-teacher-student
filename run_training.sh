#!/bin/bash
# Convenience script to run RLAIF training with Rust executable

set -e

echo "=========================================="
echo "Small LLM Code RLAIF Training Setup (Rust)"
echo "=========================================="

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
RUST_BINARY="$PROJECT_ROOT/rust_rlaif/target/release/rlaif-trainer"

# Check if Rust binary exists
if [ ! -f "$RUST_BINARY" ]; then
    echo "ERROR: Rust binary not found at: $RUST_BINARY"
    echo ""
    echo "Please build the Rust binary first:"
    echo "  cd rust_rlaif"
    echo "  cargo build --release"
    exit 1
fi

# Activate venv if it exists and isn't already activated
if [ -z "$VIRTUAL_ENV" ] && [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Check for API key
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY"
    echo "  export OPENAI_API_KEY='your-key'"
    echo "  OR"
    echo "  export ANTHROPIC_API_KEY='your-key'"
    exit 1
fi

# Create necessary directories
mkdir -p data
mkdir -p checkpoints
mkdir -p logs/tensorboard

# Generate sample data if it doesn't exist
if [ ! -f "data/train.jsonl" ]; then
    echo "Generating sample training data..."
    if command -v uv &> /dev/null; then
        uv run python scripts/utils/data_utils.py
    else
        echo "WARNING: uv not found. Skipping data generation."
        echo "  Please create data/train.jsonl manually or install uv."
    fi
fi

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "ERROR: config.yaml not found!"
    exit 1
fi

echo ""
echo "Rust binary: $RUST_BINARY"
echo "Config file: $PROJECT_ROOT/config.yaml"
echo ""
echo "Starting training..."
echo "To monitor progress, run in another terminal:"
echo "  PYTHONWARNINGS=ignore::UserWarning uv run tensorboard --logdir ./logs/tensorboard"
echo "  (or: uv run tensorboard --logdir ./logs/tensorboard 2>/dev/null)"
echo ""
echo "=========================================="
echo ""

# Set PYO3_PYTHON to use the correct Python executable (venv Python if available)
# This ensures PyO3 uses the same Python where MLX is installed
# IMPORTANT: This must be set BEFORE the binary runs, as PyO3 detects Python at startup
if [ -z "$PYO3_PYTHON" ]; then
    # First, try to use the venv Python directly
    if [ -f "$PROJECT_ROOT/.venv/bin/python" ]; then
        export PYO3_PYTHON="$PROJECT_ROOT/.venv/bin/python"
        echo "Set PYO3_PYTHON=$PYO3_PYTHON (venv Python)"
    # Try 'python' from PATH (usually points to venv if active)
    elif command -v python &> /dev/null; then
        PYTHON_EXE=$(which python)
        if [ -n "$PYTHON_EXE" ]; then
            export PYO3_PYTHON="$PYTHON_EXE"
            echo "Set PYO3_PYTHON=$PYO3_PYTHON (from PATH)"
        fi
    # Fallback to python3
    elif command -v python3 &> /dev/null; then
        PYTHON_EXE=$(which python3)
        if [ -n "$PYTHON_EXE" ]; then
            export PYO3_PYTHON="$PYTHON_EXE"
            echo "Set PYO3_PYTHON=$PYO3_PYTHON (fallback)"
        fi
    fi
fi

# Verify PYO3_PYTHON points to a Python with MLX installed
if [ -n "$PYO3_PYTHON" ] && [ -f "$PYO3_PYTHON" ]; then
    if ! "$PYO3_PYTHON" -c "import mlx_lm" 2>/dev/null; then
        echo "WARNING: PYO3_PYTHON=$PYO3_PYTHON does not have mlx_lm installed"
        echo "  Installing MLX in venv..."
        "$PYO3_PYTHON" -m pip install mlx mlx-lm --quiet 2>&1 | grep -v "already satisfied" || true
    fi
    
    # Check if binary needs to be rebuilt with this Python
    # PyO3 detects Python at build time, so we need to rebuild if PYO3_PYTHON changed
    BINARY_BUILD_TIME=$(stat -f "%m" "$RUST_BINARY" 2>/dev/null || echo "0")
    PYTHON_MTIME=$(stat -f "%m" "$PYO3_PYTHON" 2>/dev/null || echo "0")
    
    # If Python is newer than binary, suggest rebuild
    if [ "$PYTHON_MTIME" -gt "$BINARY_BUILD_TIME" ] 2>/dev/null; then
        echo "WARNING: Python executable is newer than the Rust binary."
        echo "  You may need to rebuild: cd rust_rlaif && PYO3_PYTHON=\"$PYO3_PYTHON\" cargo build --release"
    fi
fi

# Set PYTHONHOME for PyO3 if not already set
# IMPORTANT: For venvs, Python should automatically find the venv's site-packages
# We only need PYTHONHOME if Python can't find its standard library
# When using a venv, we should set PYTHONHOME to the base Python (not the venv)
if [ -z "$PYTHONHOME" ]; then
    # Use the Python from PYO3_PYTHON if set, otherwise try to detect
    PYTHON_CMD="${PYO3_PYTHON:-python}"
    if [ -n "$PYTHON_CMD" ] && command -v "$PYTHON_CMD" &> /dev/null; then
        # Get base_prefix (the actual Python installation, not the venv)
        # This is needed because venvs don't contain the full stdlib
        PYTHONHOME=$("$PYTHON_CMD" -c "import sys; print(sys.base_prefix if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix else sys.prefix)" 2>/dev/null)
    fi
    
    # Fallback to python3 if python didn't work
    if [ -z "$PYTHONHOME" ] || [ "$PYTHONHOME" = "/install" ]; then
        if command -v python3 &> /dev/null; then
            PYTHONHOME=$(python3 -c "import sys; print(sys.base_prefix if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix else sys.prefix)" 2>/dev/null)
        fi
    fi
    
    if [ -n "$PYTHONHOME" ] && [ "$PYTHONHOME" != "/install" ]; then
        export PYTHONHOME
        echo "Set PYTHONHOME=$PYTHONHOME (base Python for stdlib)"
    else
        echo "Warning: Could not detect PYTHONHOME. Python may not find its standard library."
    fi
fi

# Set DYLD_FRAMEWORK_PATH for macOS to find Python framework at runtime
# PyO3 links against @rpath/Python3.framework but needs help finding it
if [ -z "$DYLD_FRAMEWORK_PATH" ]; then
    # Try common Python framework locations
    PYTHON_FRAMEWORK_DIR=""
    for dir in "/Library/Developer/CommandLineTools/Library/Frameworks" "/Library/Frameworks" "/System/Library/Frameworks"; do
        if [ -d "$dir/Python3.framework" ]; then
            PYTHON_FRAMEWORK_DIR="$dir"
            break
        fi
    done
    
    if [ -n "$PYTHON_FRAMEWORK_DIR" ] && [ -d "$PYTHON_FRAMEWORK_DIR" ]; then
        export DYLD_FRAMEWORK_PATH="$PYTHON_FRAMEWORK_DIR"
        echo "Set DYLD_FRAMEWORK_PATH=$DYLD_FRAMEWORK_PATH for Python framework"
    else
        echo "Warning: Could not find Python framework directory. You may need to set DYLD_FRAMEWORK_PATH manually."
        echo "Try: export DYLD_FRAMEWORK_PATH=/Library/Developer/CommandLineTools/Library/Frameworks"
    fi
fi

# Verify PYO3_PYTHON is set before running
if [ -z "$PYO3_PYTHON" ]; then
    echo "ERROR: PYO3_PYTHON is not set. This should have been set automatically."
    echo "Please set it manually:"
    echo "  export PYO3_PYTHON=\"$PROJECT_ROOT/.venv/bin/python\""
    echo "  cd rust_rlaif && cargo build --release"
    exit 1
fi

# Run training with Rust binary
cd "$PROJECT_ROOT"
echo "Running with PYO3_PYTHON=$PYO3_PYTHON"
"$RUST_BINARY" --config config.yaml

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: ./checkpoints"
echo "TensorBoard logs: ./logs/tensorboard"
echo "=========================================="

