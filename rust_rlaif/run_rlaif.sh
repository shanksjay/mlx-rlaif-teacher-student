#!/bin/bash
# Wrapper script to run RLAIF trainer with proper Python environment setup

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$SCRIPT_DIR/target/release/rlaif-trainer"

# Activate venv if it exists and isn't already activated
if [ -z "$VIRTUAL_ENV" ] && [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    echo "Please build it first with: cd rust_rlaif && cargo build --release"
    exit 1
fi

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
fi

# Set PYTHONHOME if not already set
# IMPORTANT: For venvs, use base_prefix, not prefix (venvs don't have full stdlib)
# Prefer 'python' (respects venv) over 'python3' (system Python)
if [ -z "$PYTHONHOME" ]; then
    # Use the Python from PYO3_PYTHON if set, otherwise try to detect
    PYTHON_CMD="${PYO3_PYTHON:-python}"
    if command -v "$PYTHON_CMD" &> /dev/null; then
        # Get base_prefix if in venv, otherwise prefix
        PYTHONHOME=$("$PYTHON_CMD" -c "import sys; print(sys.base_prefix if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix else sys.prefix)" 2>/dev/null)
    fi
    
    # Fallback to python3 if python didn't work
    if [ -z "$PYTHONHOME" ] || [ "$PYTHONHOME" = "/install" ]; then
        PYTHONHOME=$(python3 -c "import sys; print(sys.base_prefix if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix else sys.prefix)" 2>/dev/null)
    fi
    
    if [ -n "$PYTHONHOME" ] && [ "$PYTHONHOME" != "/install" ]; then
        export PYTHONHOME
        echo "Set PYTHONHOME=$PYTHONHOME"
    else
        echo "Warning: Could not detect Python prefix. You may need to set PYTHONHOME manually."
        echo "Example: export PYTHONHOME=\$(${PYTHON_CMD:-python} -c \"import sys; print(sys.base_prefix if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix else sys.prefix)\")"
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
    fi
fi

# Change to project root and run the binary
cd "$PROJECT_ROOT"
exec "$BINARY" "$@"

