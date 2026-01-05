# Installing Rust for RLAIF Trainer

## Quick Install

Run the installation script:

```bash
./install_rust.sh
```

Or install manually:

```bash
# Install rustup (official Rust installer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add Rust to your PATH (or restart terminal)
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

## Alternative: Using Homebrew

If you prefer Homebrew:

```bash
brew install rust
```

## After Installation

Once Rust is installed, you can build the project:

```bash
cd rust_rlaif
cargo build --release
```

## Troubleshooting

If `cargo` is not found after installation:

1. **Restart your terminal** - This reloads your PATH
2. **Or manually source**: `source $HOME/.cargo/env`
3. **Or add to your shell config** (`.zshrc` or `.bashrc`):
   ```bash
   export PATH="$HOME/.cargo/bin:$PATH"
   ```

## Verify Installation

```bash
rustc --version    # Should show Rust version
cargo --version    # Should show Cargo version
```



