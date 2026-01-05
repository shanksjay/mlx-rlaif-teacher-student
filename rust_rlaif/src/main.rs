mod config;
mod models;
mod training;
mod utils;

use anyhow::Result;
use clap::Parser;
use config::RlaifConfig;
use training::RlaifTrainer;
use tracing::info;
use tracing_subscriber;

#[derive(Parser, Debug)]
#[command(name = "rlaif-trainer")]
#[command(about = "RLAIF Training for Code Generation Models", long_about = None)]
struct Args {
    /// Path to configuration file
    #[arg(short, long, default_value = "config.yaml")]
    config: String,

    /// Model name or path (overrides config)
    #[arg(short, long)]
    model: Option<String>,

    /// Path to training data file (overrides config)
    #[arg(long)]
    train_file: Option<String>,

    /// Path to evaluation data file (overrides config)
    #[arg(long)]
    eval_file: Option<String>,

    /// Enable DEBUG logging
    #[arg(short, long)]
    debug: bool,
}

// Setup Python environment before PyO3 auto-initializes
// Note: For venvs, we should NOT set PYTHONHOME to the venv path.
// Venvs don't contain the full Python stdlib - they use the base Python installation.
// IMPORTANT: PYO3_PYTHON must be set BEFORE calling prepare_freethreaded_python()
fn setup_python_env() {
    // If PYO3_PYTHON is not set, try to detect and set it
    // This ensures PyO3 uses the correct Python (venv Python if available)
    if std::env::var("PYO3_PYTHON").is_err() {
        // First, try to find .venv/bin/python in the current directory or parent directories
        let current_dir = std::env::current_dir().ok();
        let mut venv_python = None;
        
        if let Some(mut dir) = current_dir {
            // Check current directory and up to 3 levels up
            for _ in 0..4 {
                let venv_path = dir.join(".venv").join("bin").join("python");
                if venv_path.exists() {
                    if let Some(path_str) = venv_path.to_str() {
                        venv_python = Some(path_str.to_string());
                        break;
                    }
                }
                if !dir.pop() {
                    break;
                }
            }
        }
        
        // If we found a venv Python, use it
        if let Some(venv_path) = venv_python {
            std::env::set_var("PYO3_PYTHON", &venv_path);
            eprintln!("Set PYO3_PYTHON={} (detected venv)", venv_path);
        } else {
            // Try 'python' from PATH (usually points to venv if active)
            if let Ok(output) = std::process::Command::new("which")
                .arg("python")
                .output()
            {
                if let Ok(path) = String::from_utf8(output.stdout) {
                    let path = path.trim();
                    if !path.is_empty() && std::path::Path::new(path).exists() {
                        std::env::set_var("PYO3_PYTHON", path);
                        eprintln!("Set PYO3_PYTHON={} (from PATH)", path);
                    }
                }
            }
            
            // Fallback to python3 if python didn't work
            if std::env::var("PYO3_PYTHON").is_err() {
                if let Ok(output) = std::process::Command::new("which")
                    .arg("python3")
                    .output()
                {
                    if let Ok(path) = String::from_utf8(output.stdout) {
                        let path = path.trim();
                        if !path.is_empty() && std::path::Path::new(path).exists() {
                            std::env::set_var("PYO3_PYTHON", path);
                            eprintln!("Set PYO3_PYTHON={} (fallback)", path);
                        }
                    }
                }
            }
        }
    } else {
        eprintln!("Using PYO3_PYTHON={} (from environment)", std::env::var("PYO3_PYTHON").unwrap());
    }
    
    // Set PYTHONHOME if not already set
    if std::env::var("PYTHONHOME").is_err() {
        // Use the Python from PYO3_PYTHON if set, otherwise try to detect
        let python_cmd = std::env::var("PYO3_PYTHON")
            .ok()
            .filter(|p| std::path::Path::new(p).exists())
            .or_else(|| {
                // Try 'python' first
                std::process::Command::new("which")
                    .arg("python")
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|s| s.trim().to_string())
            })
            .or_else(|| {
                // Fallback to python3
                std::process::Command::new("which")
                    .arg("python3")
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|s| s.trim().to_string())
            });
        
        if let Some(python_exe) = python_cmd {
            // Get base_prefix if in venv, otherwise prefix
            if let Ok(output) = std::process::Command::new(&python_exe)
                .arg("-c")
                .arg("import sys; print(sys.base_prefix if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix else sys.prefix)")
                .output()
            {
                if let Ok(prefix_str) = String::from_utf8(output.stdout) {
                    let prefix = prefix_str.trim();
                    if !prefix.is_empty() && prefix != "/install" {
                        std::env::set_var("PYTHONHOME", prefix);
                        eprintln!("Set PYTHONHOME={} for PyO3", prefix);
                    }
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup Python environment before initializing PyO3
    // IMPORTANT: This must happen BEFORE prepare_freethreaded_python()
    setup_python_env();
    
    // Verify PYO3_PYTHON is set (PyO3 checks this at initialization)
    let pyo3_python = std::env::var("PYO3_PYTHON")
        .unwrap_or_else(|_| "not set".to_string());
    eprintln!("PyO3 will use Python from PYO3_PYTHON={}", pyo3_python);
    
    // Initialize Python runtime manually (since we removed auto-initialize)
    pyo3::prepare_freethreaded_python();
    
    let args = Args::parse();

    // Setup logging
    let log_level = if args.debug { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("rlaif_trainer={}", log_level))
        .init();

    info!("RLAIF Training for Code Generation Models");
    info!("Loading configuration from: {}", args.config);

    // Load configuration
    let mut config = RlaifConfig::load(&args.config)?;

    // Override config with CLI arguments
    if let Some(model) = args.model {
        config.base_model = model;
    }
    if let Some(train_file) = args.train_file {
        config.train_file = train_file;
    }
    if let Some(eval_file) = args.eval_file {
        config.eval_file = Some(eval_file);
    }

    // Initialize trainer
    let trainer = RlaifTrainer::new(config).await?;

    // Load datasets
    let train_dataset = utils::load_dataset(&trainer.config.train_file)?;
    let eval_dataset = trainer.config.eval_file.as_ref()
        .and_then(|f| utils::load_dataset(f).ok());

    // Start training
    info!("Starting training...");
    trainer.train(train_dataset, eval_dataset).await?;

    info!("Training completed successfully!");
    Ok(())
}

