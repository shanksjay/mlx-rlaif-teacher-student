#!/bin/bash
# Install Rust and Cargo using rustup

echo "Installing Rust and Cargo..."

# Check if rustup is already installed
if command -v rustc &> /dev/null; then
    echo "Rust is already installed: $(rustc --version)"
    exit 0
fi

# Install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Source cargo environment
source "$HOME/.cargo/env"

# Verify installation
if command -v cargo &> /dev/null; then
    echo "✓ Rust installed successfully!"
    echo "  Rust version: $(rustc --version)"
    echo "  Cargo version: $(cargo --version)"
    echo ""
    echo "Note: You may need to restart your terminal or run:"
    echo "  source $HOME/.cargo/env"
else
    echo "✗ Installation may have failed. Please check the output above."
    exit 1
fi



