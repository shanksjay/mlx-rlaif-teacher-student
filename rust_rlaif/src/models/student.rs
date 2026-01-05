use anyhow::Result;
use tokenizers::Tokenizer;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::models::mlx_generator::MlxGenerator;
use crate::models::mlx_trainer::MlxTrainer;

/// Student model using MLX for both forward and backward passes:
/// - MLX generator for fast forward passes (generation)
/// - MLX trainer for backward passes and optimizer updates (training)
pub struct StudentModel {
    // MLX generator for fast forward passes (generation)
    pub(crate) mlx_generator: Option<Arc<Mutex<MlxGenerator>>>,
    // MLX trainer for backward passes and optimizer updates (training)
    pub(crate) mlx_trainer: Option<Arc<Mutex<MlxTrainer>>>,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) model_path: String,  // Used for error messages and logging
}

impl StudentModel {
    /// Load tokenizer from HuggingFace model ID or create minimal fallback
    fn load_tokenizer_from_hf(model_id: &str) -> Result<Tokenizer> {
        // Check if model_id looks like a HuggingFace model ID (contains '/')
        if model_id.contains('/') && !std::path::Path::new(model_id).exists() {
            // Try to load from HuggingFace cache
            // The tokenizers crate doesn't have from_pretrained, so we need to use
            // the HuggingFace cache directory structure
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            let hf_cache = std::path::Path::new(&home)
                .join(".cache")
                .join("huggingface")
                .join("hub");
            
            // Try to find tokenizer.json in HuggingFace cache
            // Model ID format: "org/model" -> "models--org--model"
            let cache_model_dir = model_id.replace('/', "--");
            
            // Search in all possible snapshot directories
            let models_dir = hf_cache.join(format!("models--{}", cache_model_dir));
            if models_dir.exists() {
                // Look for snapshots directory
                let snapshots_dir = models_dir.join("snapshots");
                if snapshots_dir.exists() {
                    // Try each snapshot directory (usually there's a hash-named directory)
                    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                        for entry in entries.flatten() {
                            let tokenizer_path = entry.path().join("tokenizer.json");
                            if tokenizer_path.exists() {
                                info!("Loading tokenizer from HuggingFace cache: {}", tokenizer_path.display());
                                match Tokenizer::from_file(&tokenizer_path) {
                                    Ok(tok) => {
                                        info!("✓ Tokenizer loaded from HuggingFace cache");
                                        return Ok(tok);
                                    }
                                    Err(e) => {
                                        warn!("Failed to load tokenizer from cache path {}: {}", tokenizer_path.display(), e);
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Also try directly in models directory
                let direct_path = models_dir.join("tokenizer.json");
                if direct_path.exists() {
                    info!("Loading tokenizer from HuggingFace cache: {}", direct_path.display());
                    match Tokenizer::from_file(&direct_path) {
                        Ok(tok) => {
                            info!("✓ Tokenizer loaded from HuggingFace cache");
                            return Ok(tok);
                        }
                        Err(e) => {
                            warn!("Failed to load tokenizer from cache path {}: {}", direct_path.display(), e);
                        }
                    }
                }
            }
            
            warn!("Tokenizer not found in HuggingFace cache for {}. You may need to download it first using Python: from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('{}').save_pretrained('./tokenizer')", model_id, model_id);
        }
        
        // Fallback: create minimal tokenizer
        warn!("Creating minimal tokenizer as fallback. For proper tokenization, ensure tokenizer.json exists in MLX model path or HuggingFace cache.");
        Ok(Tokenizer::new(tokenizers::models::bpe::BPE::default()))
    }
    
    /// Load student model with MLX for both generation and training
    pub async fn load<P: AsRef<Path>>(
        model_path: P,
        _device_str: &str,
        mlx_model_path: Option<&str>,
        _coreml_model_path: Option<&str>,
    ) -> Result<Self> {
        let model_path_str = model_path.as_ref().to_string_lossy().to_string();
        info!("Loading student model from: {}", model_path_str);
        
        // Load MLX generator for fast forward passes (generation)
        let mlx_generator = if let Some(mlx_path) = mlx_model_path {
            let mlx_path = std::path::Path::new(mlx_path);
            if mlx_path.exists() {
                info!("Loading MLX generator from: {}", mlx_path.display());
                match MlxGenerator::new(mlx_path).await {
                    Ok(gen) => {
                        info!("MLX generator loaded successfully");
                        Some(Arc::new(Mutex::new(gen)))
                    }
                    Err(e) => {
                        warn!("Failed to load MLX generator: {}. Generation will use API fallback.", e);
                        None
                    }
                }
            } else {
                warn!("MLX model path does not exist: {}. Generation will use API fallback.", mlx_path.display());
                None
            }
        } else {
            None
        };
        
        // Load MLX trainer for backward passes and optimizer updates (training)
        // Use the same MLX model path for training
        // MLX trainer should be enabled by default when mlx_model_path is provided
        let mlx_trainer = if let Some(mlx_path) = mlx_model_path {
            let mlx_path = std::path::Path::new(mlx_path);
            info!("🔍 MLX trainer path check: input='{}', exists={}", mlx_path.display(), mlx_path.exists());
            if mlx_path.exists() {
                let trainer_path = {
                    let mut path = mlx_path.to_path_buf();
                    
                    // Check if path contains q4 or q8 (case-insensitive)
                    // Check both the full path string and the file_name
                    let path_str = mlx_path.to_string_lossy().to_lowercase();
                    let name_lower = mlx_path.file_name()
                        .and_then(|n| n.to_str())
                        .map(|s| s.to_ascii_lowercase())
                        .unwrap_or_default();
                    
                    debug!("🔍 Path analysis: path_str='{}', name_lower='{}'", path_str, name_lower);
                    
                    // More robust detection: check if path contains /q4 or /q8, or if filename is q4/q8
                    let is_quantized = path_str.contains("/q4") || path_str.contains("/q8") ||
                                       path_str.contains("\\q4") || path_str.contains("\\q8") ||
                                       path_str.ends_with("/q4") || path_str.ends_with("/q8") ||
                                       path_str.ends_with("\\q4") || path_str.ends_with("\\q8") ||
                                       path_str == "q4" || path_str == "q8" ||
                                       name_lower == "q4" || name_lower == "q8";
                    
                    info!("🔍 Quantized detection result: is_quantized={}, path_str='{}', name_lower='{}'", 
                        is_quantized, path_str, name_lower);
                    
                    if is_quantized {
                        info!("🔍 Detected quantized model path: {} (filename: '{}')", 
                            mlx_path.display(), name_lower);
                        
                        // Try to canonicalize for base path resolution, but fall back to original if it fails
                        let path_for_base = mlx_path.canonicalize()
                            .unwrap_or_else(|_| {
                                warn!("Could not canonicalize path {}, using as-is", mlx_path.display());
                                mlx_path.to_path_buf()
                            });
                        
                        info!("🔍 Checking for unquantized base model at parent of {}...", path_for_base.display());
                        match ensure_unquantized_base(&path_for_base) {
                            Ok(base_path) => {
                                info!(
                                    "✅ Using unquantized base model for training: {} (generation still uses {})",
                                    base_path.display(),
                                    mlx_path.display()
                                );
                                path = base_path;
                            }
                            Err(e) => {
                                error!(
                                    "❌ Failed to create unquantized base model from {}: {}",
                                    path_for_base.display(), e
                                );
                                error!("⚠️  Training will use quantized path, which WILL FAIL during optimizer updates.");
                                error!("⚠️  The optimizer cannot update quantized (uint32) weights - they will be corrupted.");
                                error!("💡 To fix: Ensure mlx_lm is installed and the base model can be created.");
                                // Continue with quantized path (will definitely fail)
                            }
                        }
                    } else {
                        debug!("Model path '{}' (filename: '{}') is not quantized (q4/q8), using as-is for training", 
                            mlx_path.display(), name_lower);
                    }
                    path
                };
                
                // Safety check: if we detected a quantized path but didn't switch to base, warn loudly
                let final_path_str = trainer_path.to_string_lossy().to_lowercase();
                let final_name = trainer_path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.to_ascii_lowercase())
                    .unwrap_or_default();
                if (final_path_str.contains("/q4") || final_path_str.contains("/q8") ||
                    final_name == "q4" || final_name == "q8") && 
                   !final_path_str.contains("/base") && final_name != "base" {
                    error!("🚨 CRITICAL: Trainer path is still quantized: {}", trainer_path.display());
                    error!("🚨 This will cause training to fail! The optimizer cannot update quantized weights.");
                    error!("🚨 Expected base model path but got: {}", trainer_path.display());
                    return Err(anyhow::anyhow!(
                        "Cannot train on quantized model: {}. Please ensure base model exists or dequantization succeeds.",
                        trainer_path.display()
                    ));
                }
                
                info!("Loading MLX trainer from: {}", trainer_path.display());
                match MlxTrainer::new(&trainer_path, true).await {
                    Ok(trainer) => {
                        info!("MLX trainer loaded successfully");
                        Some(Arc::new(Mutex::new(trainer)))
                    }
                    Err(e) => {
                        error!("Failed to load MLX trainer from {}: {}", mlx_path.display(), e);
                        error!("Error chain: {:?}", e);
                        // Try to get the root cause
                        if let Some(source) = e.source() {
                            error!("Root cause: {}", source);
                        }
                        error!("Training will use simulated backprop instead of MLX training.");
                        error!("To fix: Ensure MLX is installed (pip install mlx mlx-lm) and Python environment is properly configured.");
                        None
                    }
                }
            } else {
                warn!("MLX model path does not exist: {}. MLX trainer will not be available.", mlx_path.display());
                None
            }
        } else {
            warn!("No mlx_model_path configured. MLX trainer will not be available. Set hardware.mlx_model_path in config.yaml to enable MLX training.");
            None
        };
        
        // Load tokenizer
        // Priority: 1. MLX model path (contains tokenizer.json), 2. Local path, 3. HuggingFace cache, 4. Minimal fallback
        let tokenizer = {
            // First, try MLX model path (most reliable - MLX models include tokenizers)
            let mlx_tokenizer_opt = if let Some(mlx_path) = mlx_model_path {
                let mlx_path = std::path::Path::new(mlx_path);
                let mlx_tokenizer_path = mlx_path.join("tokenizer.json");
                if mlx_tokenizer_path.exists() {
                    info!("Loading tokenizer from MLX model path: {}", mlx_tokenizer_path.display());
                    match Tokenizer::from_file(&mlx_tokenizer_path) {
                        Ok(tok) => {
                            info!("✓ Tokenizer loaded from MLX model path");
                            Some(tok)
                        }
                        Err(e) => {
                            let error_msg = format!("{}", e);
                            // Check file size to see if it might be corrupted
                            let file_size = std::fs::metadata(&mlx_tokenizer_path)
                                .map(|m| m.len())
                                .unwrap_or(0);
                            
                            if error_msg.contains("ModelWrapper") || error_msg.contains("variant") {
                                warn!(
                                    "Tokenizer file at {} appears to be incompatible or corrupted (size: {} bytes). \
                                    This may happen if the tokenizer.json was generated with a different tokenizers version. \
                                    Falling back to HuggingFace cache...",
                                    mlx_tokenizer_path.display(),
                                    file_size
                                );
                            } else {
                                warn!(
                                    "Failed to load tokenizer from MLX path {} (size: {} bytes): {}. \
                                    Trying HuggingFace cache...",
                                    mlx_tokenizer_path.display(),
                                    file_size,
                                    e
                                );
                            }
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };
            
            // If MLX tokenizer loaded, use it
            if let Some(tok) = mlx_tokenizer_opt {
                tok
            } else {
                // Try local path (if model_path is a local directory)
                let local_tokenizer_path = model_path.as_ref().join("tokenizer.json");
                if local_tokenizer_path.exists() {
                    info!("Loading tokenizer from local path: {}", local_tokenizer_path.display());
                    match Tokenizer::from_file(&local_tokenizer_path) {
                        Ok(tok) => {
                            info!("✓ Tokenizer loaded from local path");
                            tok
                        }
                        Err(e) => {
                            warn!("Failed to load tokenizer from local path: {}. Trying HuggingFace cache...", e);
                            Self::load_tokenizer_from_hf(&model_path_str)?
                        }
                    }
                } else {
                    // Try HuggingFace cache or fallback
                    Self::load_tokenizer_from_hf(&model_path_str)?
                }
            }
        };

        Ok(Self {
            mlx_generator,
            mlx_trainer,
            tokenizer,
            model_path: model_path_str,
        })
    }

    /// Training step: forward + backward + optimizer update using MLX
    pub async fn train_step(
        &self,
        input_ids: &[Vec<u32>],
        advantages: &[f64],
        kl_penalty: f64,
        reward_weight: f64,
        max_grad_norm: f64,
        learning_rate: f64,
        gradient_accumulation_steps: usize,
        accumulation_step: usize,
    ) -> Result<(f64, f64)> {
        let trainer = self.mlx_trainer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("MLX trainer not loaded for training"))?;
        let trainer = trainer.lock().await;
        
        trainer.train_step(
            input_ids,
            advantages,
            kl_penalty,
            reward_weight,
            max_grad_norm,
            learning_rate,
            gradient_accumulation_steps,
            accumulation_step,
        ).await
    }

    /// Generate text using MLX (fast forward pass for generation)
    pub async fn generate(&self, prompt: &str, max_tokens: usize, temperature: f64) -> Result<String> {
        // Use MLX generator if available (fast forward pass)
        if let Some(mlx_gen) = &self.mlx_generator {
            let mlx = mlx_gen.lock().await;
            // Format prompt similar to Python version
            let formatted_prompt = format!("Write high-quality code:\n\n{prompt}\n\nCode:");
            
            // Generate text (MLX returns prompt + generated text)
            let full_text = mlx.generate(
                &formatted_prompt,
                max_tokens,
                temperature,
                0.9, // top_p
                50,  // top_k
            ).await?;
            
            // Extract just the generated portion (remove prompt)
            // MLX returns the full text including the prompt, so we need to strip it
            let generated = crate::models::code_extractor::extract_generated_code(&full_text, &formatted_prompt);
            
            // Extract the first complete code block to prevent repetition
            // This ensures we get complete, valid code blocks
            let code = crate::models::code_extractor::extract_first_code_block(&generated);
            
            // If extraction resulted in empty or very short code, use the generated text as-is
            // (might be code without markdown fences)
            if code.len() < 10 && generated.len() > 10 {
                Ok(generated)
            } else {
                Ok(code)
            }
        } else {
            anyhow::bail!(
                "MLX generator not available for fast forward pass (model: {}). \
                Cannot generate. Please provide mlx_model_path in config.",
                self.model_path
            )
        }
    }
}

fn ensure_unquantized_base(q_path: &Path) -> Result<PathBuf> {
    // Canonicalize the path to ensure consistent path handling
    let q_path_canonical = q_path.canonicalize()
        .map_err(|e| anyhow::anyhow!("Failed to canonicalize quantized model path {}: {}", q_path.display(), e))?;
    
    // Get parent directory and create base path
    let parent = q_path_canonical.parent()
        .ok_or_else(|| anyhow::anyhow!("Quantized model path has no parent directory: {}", q_path_canonical.display()))?;
    let base_path = parent.join("base");
    
    debug!("Checking for base model: quantized={}, base={}", 
        q_path_canonical.display(), base_path.display());
    
    if base_path.exists() {
        info!("✓ Unquantized base model already exists at {}", base_path.display());
        return Ok(base_path);
    }

    info!("Unquantized base model not found at {}. Attempting to dequantize {} to {}...",
        base_path.display(), q_path_canonical.display(), base_path.display());

    Python::with_gil(|py| -> Result<()> {
        // Use the same API pattern as export_fused_mlx_model
        let script = format!(
            r#"
import os
from pathlib import Path
from mlx_lm.utils import load, save
from mlx_lm.convert import dequantize_model

src = Path(r"{}")
dest = Path(r"{}")

if dest.exists():
    print(f"Base model already exists at {{dest}}")
else:
    print(f"Loading quantized model from {{src}}...")
    model, tokenizer, config = load(str(src), return_config=True)
    
    print(f"Dequantizing model...")
    model = dequantize_model(model)
    
    # Remove quantization config
    if "quantization" in config:
        del config["quantization"]
    if "quantization_config" in config:
        del config["quantization_config"]
    
    print(f"Saving dequantized model to {{dest}}...")
    dest.mkdir(parents=True, exist_ok=True)
    save(dest, str(src), model, tokenizer, config, donate_model=False)
    print(f"✓ Successfully dequantized model to {{dest}}")
"#,
            q_path_canonical.to_string_lossy().replace('\\', "\\\\"),
            base_path.to_string_lossy().replace('\\', "\\\\")
        );

        #[allow(deprecated)]
        let builtins = py.import("builtins")
            .map_err(|e| anyhow::anyhow!("Failed to import builtins: {}", e))?;
        let compile_fn = builtins.getattr("compile")
            .map_err(|e| anyhow::anyhow!("Failed to get compile function: {}", e))?;
        let code_obj = compile_fn.call1((script.as_str(), "<string>", "exec"))
            .map_err(|e| anyhow::anyhow!("Failed to compile dequantization script: {}", e))?;

        #[allow(deprecated)]
        let ns = PyDict::new(py);
        let exec_fn = builtins.getattr("exec")
            .map_err(|e| anyhow::anyhow!("Failed to get exec function: {}", e))?;
        
        exec_fn.call1((code_obj, ns))
            .map_err(|e| {
                // Try to extract Python error details
                let error_msg = format!("{}", e);
                anyhow::anyhow!("Failed to dequantize model from {} to {}: {}", 
                    q_path_canonical.display(), base_path.display(), error_msg)
            })?;

        Ok(())
    })?;

    // Verify the base path was created
    if !base_path.exists() {
        return Err(anyhow::anyhow!(
            "Dequantization completed but base model path does not exist: {}", 
            base_path.display()
        ));
    }

    info!("✓ Successfully dequantized {} to {}", q_path.display(), base_path.display());
    Ok(base_path)
}

