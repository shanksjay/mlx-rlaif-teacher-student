use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

/// MLX Trainer for backward passes and optimizer updates
/// Uses PyO3 to call MLX directly from Rust (no subprocess overhead)
pub struct MlxTrainer {
    model: PyObject,
    optimizer: Arc<Mutex<Option<PyObject>>>,
    model_path: String,
    use_lora: bool,
    // LoRA config (used for saving MLX-compatible adapter dirs)
    lora_rank: Option<usize>,
    lora_alpha: Option<usize>,
    lora_dropout: Option<f64>,
    lora_num_layers: Option<usize>,
}

impl MlxTrainer {
    /// Load MLX model and initialize trainer
    pub async fn new<P: AsRef<Path>>(model_path: P, use_lora: bool) -> Result<Self> {
        // QLoRA approach:
        // - Allow loading a quantized base model (q4/q8) for memory/throughput benefits.
        // - Inject LoRA layers on top of quantized Linear/Embedding modules.
        // - During updates, only apply gradients to trainable (LoRA) parameters.
        //
        // This avoids corrupting quantized base weights, which can otherwise lead to:
        //   ValueError: [dequantize] The matrix should be given as a uint32
        let model_path_str = model_path.as_ref().to_string_lossy().to_string();

        // LoRA hyperparams (env overrides; used both for injection and adapter config saving)
        let (lora_rank, lora_alpha, lora_dropout, lora_num_layers) = if use_lora {
            let rank: usize = std::env::var("MLX_LORA_RANK")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(16);
            let alpha: usize = std::env::var("MLX_LORA_ALPHA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32);
            let dropout: f64 = std::env::var("MLX_LORA_DROPOUT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.05);
            let num_layers: usize = std::env::var("MLX_LORA_NUM_LAYERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8);
            (Some(rank), Some(alpha), Some(dropout), Some(num_layers))
        } else {
            (None, None, None, None)
        };

        info!(
            "Loading MLX model for training: {}{}",
            model_path_str,
            if use_lora {
                " (LoRA adapters will be injected before training)"
            } else {
                ""
            }
        );

        // Initialize Python if not already initialized
        pyo3::prepare_freethreaded_python();
        
        // Acquire Python GIL and load model
        let model = Python::with_gil(|py| -> Result<PyObject> {
            // Get current Python executable for error messages
            // Note: sys.executable may point to the embedding binary when Python is embedded
            // So we also check PYO3_PYTHON and sys.prefix to get the actual Python
            let current_python = std::env::var("PYO3_PYTHON")
                .unwrap_or_else(|_| {
                    py.run_bound("import sys", None, None)
                        .ok()
                        .and_then(|_| py.eval_bound("sys.executable", None, None).ok())
                        .and_then(|obj| obj.extract::<String>().ok())
                        .unwrap_or_else(|| "unknown".to_string())
                });
            
            // Also get sys.prefix to show where Python thinks its packages are
            let python_prefix = py.run_bound("import sys", None, None)
                .ok()
                .and_then(|_| py.eval_bound("sys.prefix", None, None).ok())
                .and_then(|obj| obj.extract::<String>().ok())
                .unwrap_or_else(|| "unknown".to_string());
            
            debug!("PyO3 using Python: {} (prefix: {})", current_python, python_prefix);
            debug!("PYO3_PYTHON env var: {:?}", std::env::var("PYO3_PYTHON"));
            
            // When Python is embedded, it doesn't automatically activate venvs
            // We need to manually add the venv's site-packages to sys.path
            if current_python.contains(".venv") || current_python.contains("venv") {
                // Use Python to dynamically detect the venv site-packages path
                // This is more reliable than hardcoding Python version
                // Use PYO3_PYTHON env var as it's more reliable than sys.executable when embedded
                let activate_venv_code = r#"
import sys
import os

# Get Python executable path from PYO3_PYTHON env var (more reliable when embedded)
python_exe = os.environ.get('PYO3_PYTHON', '')
if not python_exe:
    # Fallback to sys.executable
    python_exe = getattr(sys, 'executable', '') or (sys.argv[0] if sys.argv else '')

# Extract venv path from Python executable
if python_exe and ('.venv' in python_exe or 'venv' in python_exe):
    venv_bin = os.path.dirname(os.path.abspath(python_exe))
    venv_root = os.path.dirname(venv_bin)
    
    # Try to find site-packages in the venv
    # Check common locations: lib/pythonX.Y/site-packages
    major, minor = sys.version_info[:2]
    possible_paths = [
        os.path.join(venv_root, 'lib', f'python{major}.{minor}', 'site-packages'),
        os.path.join(venv_root, 'lib', f'python{major}', 'site-packages'),
        os.path.join(venv_root, 'lib', 'site-packages'),
    ]
    
    for site_packages in possible_paths:
        if os.path.exists(site_packages):
            if site_packages not in sys.path:
                sys.path.insert(0, site_packages)
            break
"#;
                
                if let Err(e) = py.run_bound(&activate_venv_code, None, None) {
                    warn!("Failed to activate venv site-packages: {}", e);
                } else {
                    debug!("Activated venv site-packages for embedded Python");
                }
            }
            
            #[allow(deprecated)] // TODO: Migrate to PyModule::import_bound in future PyO3 version
            let mlx_lm = match PyModule::import(py, "mlx_lm") {
                Ok(m) => m,
                Err(e) => {
                    // Get Python path info for debugging
                    let python_paths = py.run_bound("import sys", None, None)
                        .ok()
                        .and_then(|_| {
                            py.eval_bound("';'.join(sys.path)", None, None).ok()
                                .and_then(|obj| obj.extract::<String>().ok())
                        })
                        .unwrap_or_else(|| "unknown".to_string());
                    
                    error!("Failed to import mlx_lm module. Error: {}", e);
                    error!("Make sure MLX is installed in your Python environment:");
                    error!("  pip install mlx mlx-lm");
                    error!("");
                    error!("Python environment info:");
                    error!("  PYO3_PYTHON: {:?}", std::env::var("PYO3_PYTHON"));
                    error!("  PYTHONHOME: {:?}", std::env::var("PYTHONHOME"));
                    error!("  Python executable: {}", current_python);
                    error!("  Python prefix: {}", python_prefix);
                    error!("  Python sys.path: {}", python_paths);
                    error!("");
                    error!("To fix:");
                    error!("  1. Ensure venv is activated: source .venv/bin/activate");
                    error!("  2. Verify MLX is installed: python -c 'import mlx_lm; print(\"OK\")'");
                    error!("  3. Set PYO3_PYTHON before running: export PYO3_PYTHON=\"$PWD/.venv/bin/python\"");
                    error!("  4. Rebuild if needed: cd rust_rlaif && cargo build --release");
                    return Err(anyhow::anyhow!("Failed to import mlx_lm. Python: {} (prefix: {}). Error: {}", current_python, python_prefix, e));
                }
            };
            
            let load_fn = mlx_lm.getattr("load")
                .map_err(|e| anyhow::anyhow!("Failed to get mlx_lm.load function: {}", e))?;
            
            // Load model: model, tokenizer = mlx_lm.load(model_path)
            let args = (model_path_str.as_str(),);
            let result = load_fn
                .call1(args)
                .map_err(|e| anyhow::anyhow!("Failed to call mlx_lm.load: {}", e))?;
            
            // Extract model/tokenizer
            let model_tuple = result.downcast::<PyTuple>()
                .map_err(|e| anyhow::anyhow!("mlx_lm.load should return (model, tokenizer), got: {}", e))?;
            let model = model_tuple
                .get_item(0)
                .map_err(|e| anyhow::anyhow!("Failed to get model from load result: {}", e))?;
            
            // Get the underlying model (mlx_lm wraps it in a Model class)
            let model_obj = if model.hasattr("model")? {
                model.getattr("model")
                    .map_err(|e| anyhow::anyhow!("Failed to get model.model attribute: {}", e))?
                    .to_object(py)
            } else {
                model.to_object(py)
            };

            // If training on a quantized base, inject LoRA layers.
            if use_lora {
                let rank = lora_rank.unwrap_or(16);
                let alpha = lora_alpha.unwrap_or(32);
                let dropout = lora_dropout.unwrap_or(0.05);
                let num_layers = lora_num_layers.unwrap_or(8);

                let scale = (alpha as f64) / (rank as f64);
                info!(
                    "Applying LoRA adapters (rank={}, alpha={}, scale={:.4}, dropout={}, num_layers={})",
                    rank, alpha, scale, dropout, num_layers
                );

                #[allow(deprecated)] // TODO: Migrate to PyModule::import_bound in future PyO3 version
                let lora_mod = PyModule::import(py, "mlx_lm.lora")
                    .map_err(|e| anyhow::anyhow!("Failed to import mlx_lm.lora: {}", e))?;
                let linear_to_lora_layers = lora_mod.getattr("linear_to_lora_layers")
                    .map_err(|e| anyhow::anyhow!("Failed to get mlx_lm.lora.linear_to_lora_layers: {}", e))?;

                #[allow(deprecated)] // TODO: Migrate to PyDict::new_bound in future PyO3 version
                let lora_cfg = PyDict::new(py);
                lora_cfg.set_item("rank", rank)
                    .map_err(|e| anyhow::anyhow!("Failed to set lora config rank: {}", e))?;
                lora_cfg.set_item("scale", scale)
                    .map_err(|e| anyhow::anyhow!("Failed to set lora config scale: {}", e))?;
                lora_cfg.set_item("dropout", dropout)
                    .map_err(|e| anyhow::anyhow!("Failed to set lora config dropout: {}", e))?;

                let model_bound = model_obj.bind(py);
                linear_to_lora_layers.call1((model_bound, num_layers, lora_cfg, false))
                    .map_err(|e| anyhow::anyhow!("Failed to apply LoRA layers to model: {}", e))?;
            }

            info!("MLX model loaded successfully");
            Ok(model_obj)
        })?;
        
        Ok(Self {
            model,
            optimizer: Arc::new(Mutex::new(None)),
            model_path: model_path_str,
            use_lora,
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_num_layers,
        })
    }

    /// Save adapters in an MLX-compatible adapter directory:
    /// - `adapter_config.json`
    /// - `adapters.safetensors`
    ///
    /// For QLoRA runs, `model.trainable_parameters()` corresponds to LoRA params only.
    pub async fn save_adapters<P: AsRef<Path>>(&self, adapter_dir: P) -> Result<()> {
        let adapter_dir_str = adapter_dir.as_ref().to_string_lossy().to_string();

        Python::with_gil(|py| -> Result<()> {
            let trainable = self.model.bind(py).call_method0("trainable_parameters")
                .map_err(|e| anyhow::anyhow!("Failed to call model.trainable_parameters(): {}", e))?;

            let rank = self.lora_rank.unwrap_or(16);
            let alpha = self.lora_alpha.unwrap_or(32);
            let dropout = self.lora_dropout.unwrap_or(0.05);
            let num_layers = self.lora_num_layers.unwrap_or(8);
            let scale = (alpha as f64) / (rank as f64);

            let save_code = format!(
                r#"
import os, json
import mlx.core as mx
from mlx.utils import tree_flatten

adapter_dir = r"{}"
os.makedirs(adapter_dir, exist_ok=True)

# Adapter config expected by mlx_lm.lora.load_adapters()
cfg = {{
  "fine_tune_type": "lora",
  "num_layers": {},
  "lora_parameters": {{
    "rank": {},
    "scale": {},
    "dropout": {}
  }}
}}
with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
  json.dump(cfg, f, indent=2)

adapter_weights = dict(tree_flatten(trainable))
mx.save_safetensors(os.path.join(adapter_dir, "adapters.safetensors"), adapter_weights)
"#,
                adapter_dir_str.replace('\\', "\\\\"),
                num_layers,
                rank,
                scale,
                dropout
            );

            #[allow(deprecated)] // TODO: Migrate to Python::import_bound in future PyO3 version
            let builtins = py.import("builtins")
                .map_err(|e| anyhow::anyhow!("Failed to import builtins: {}", e))?;
            let exec_fn = builtins
                .getattr("exec")
                .map_err(|e| anyhow::anyhow!("Failed to get exec function: {}", e))?;
            #[allow(deprecated)] // TODO: Migrate to PyDict::new_bound in future PyO3 version
            let ns = PyDict::new(py);
            ns.set_item("trainable", trainable)
                .map_err(|e| anyhow::anyhow!("Failed to set trainable in namespace: {}", e))?;
            exec_fn
                .call1((save_code.as_str(), ns))
                .map_err(|e| anyhow::anyhow!("Failed to save adapters to {}: {}", adapter_dir_str, e))?;

            Ok(())
        })
    }

    /// Export a fused MLX model (base + adapters) to `output_dir`.
    ///
    /// - Loads base model from `self.model_path`
    /// - Loads adapters from `adapter_dir` (expects MLX format: adapter_config.json + adapters.safetensors)
    /// - Fuses LoRA modules (calls `.fuse()` where available)
    /// - Optionally dequantizes/re-quantizes based on `quantization`:
    ///     - `q4_bit` -> 4-bit affine, group_size defaults to config quantization or 64
    ///     - `q8_bit` -> 8-bit affine, group_size defaults to config quantization or 64
    pub async fn export_fused_mlx_model<P: AsRef<Path>>(
        &self,
        adapter_dir: P,
        output_dir: P,
        quantization: Option<&str>,
    ) -> Result<()> {
        let model_path = self.model_path.clone();
        let adapter_dir_str = adapter_dir.as_ref().to_string_lossy().to_string();
        let output_dir_str = output_dir.as_ref().to_string_lossy().to_string();
        let quant = quantization.unwrap_or("").to_string();

        Python::with_gil(|py| -> Result<()> {
            let code = r#"
import os
from pathlib import Path
from mlx.utils import tree_unflatten

from mlx_lm.utils import load, save, dequantize_model

def _parse_quant(q):
    q = (q or "").strip().lower()
    if not q or q in ("none", "null", "false"):
        return None
    if q == "q4_bit":
        return {"bits": 4}
    if q == "q8_bit":
        return {"bits": 8}
    # Unknown value: ignore (keep as-is)
    return None

def _current_bits(cfg):
    q = cfg.get("quantization") or cfg.get("quantization_config") or None
    if isinstance(q, dict) and "bits" in q:
        try:
            return int(q["bits"])
        except Exception:
            return None
    return None

model_path = MODEL_PATH
adapter_path = ADAPTER_DIR
out_path = Path(OUT_DIR)
out_path.mkdir(parents=True, exist_ok=True)

# Load base + adapters
model, tokenizer, config = load(model_path, adapter_path=adapter_path, return_config=True)

# Fuse LoRA modules into base
fused = [(n, m.fuse(dequantize=False)) for n, m in model.named_modules() if hasattr(m, "fuse")]
if fused:
    model.update_modules(tree_unflatten(fused))

want = _parse_quant(QUANT)
if want is not None:
    want_bits = int(want["bits"])
    have_bits = _current_bits(config)
    if have_bits != want_bits:
        # Re-quantize to requested bits:
        # 1) dequantize model weights
        model = dequantize_model(model)
        config.pop("quantization", None)
        config.pop("quantization_config", None)

        # 2) quantize
        from mlx_lm.convert import quantize_model
        # Try to preserve group_size/mode if available
        group_size = 64
        mode = "affine"
        qcfg = config.get("quantization") or config.get("quantization_config") or None
        if isinstance(qcfg, dict):
            group_size = int(qcfg.get("group_size", group_size))
            mode = str(qcfg.get("mode", mode))
        model, config = quantize_model(model, config, group_size=group_size, bits=want_bits, mode=mode)

# Save fused model
save(out_path, model_path, model, tokenizer, config, donate_model=False)
"#;

            // Execute with injected vars
            #[allow(deprecated)] // TODO: Migrate to Python::import_bound in future PyO3 version
            let builtins = py.import("builtins")
                .map_err(|e| anyhow::anyhow!("Failed to import builtins: {}", e))?;
            let exec_fn = builtins.getattr("exec")
                .map_err(|e| anyhow::anyhow!("Failed to get exec function: {}", e))?;

            #[allow(deprecated)] // TODO: Migrate to PyDict::new_bound in future PyO3 version
            let ns = PyDict::new(py);
            ns.set_item("MODEL_PATH", model_path.as_str())
                .map_err(|e| anyhow::anyhow!("Failed to set MODEL_PATH: {}", e))?;
            ns.set_item("ADAPTER_DIR", adapter_dir_str.as_str())
                .map_err(|e| anyhow::anyhow!("Failed to set ADAPTER_DIR: {}", e))?;
            ns.set_item("OUT_DIR", output_dir_str.as_str())
                .map_err(|e| anyhow::anyhow!("Failed to set OUT_DIR: {}", e))?;
            ns.set_item("QUANT", quant.as_str())
                .map_err(|e| anyhow::anyhow!("Failed to set QUANT: {}", e))?;

            exec_fn
                .call1((code, ns))
                .map_err(|e| anyhow::anyhow!("Failed to export fused MLX model: {}", e))?;

            Ok(())
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
        let start_time = std::time::Instant::now();
        
        // Lock optimizer before entering Python GIL context (to avoid blocking inside GIL)
        let optimizer_needs_init = {
            let optimizer_guard = self.optimizer.lock().await;
            optimizer_guard.is_none()
        };
        
        // Initialize optimizer if needed (outside GIL to avoid blocking issues)
        if optimizer_needs_init {
            let optimizer_obj = Python::with_gil(|py| -> Result<PyObject> {
                // Use Python code to create optimizer with keyword arguments
                let optimizer_code = format!(
                    r#"
import mlx.optimizers as optim
optimizer = optim.Adam(learning_rate={})
"#,
                    learning_rate
                );
                
                // Compile and execute the optimizer creation code
                #[allow(deprecated)] // TODO: Migrate to Python::import_bound in future PyO3 version
                let builtins = py.import("builtins")
                    .map_err(|e| anyhow::anyhow!("Failed to import builtins: {}", e))?;
                let compile_fn = builtins.getattr("compile")
                    .map_err(|e| anyhow::anyhow!("Failed to get compile function: {}", e))?;
                let code_obj = compile_fn.call1((optimizer_code.as_str(), "<string>", "exec"))
                    .map_err(|e| anyhow::anyhow!("Failed to compile optimizer code: {}", e))?;
                
                #[allow(deprecated)] // TODO: Migrate to PyDict::new_bound in future PyO3 version
                let namespace = pyo3::types::PyDict::new(py);
                let exec_fn = builtins.getattr("exec")
                    .map_err(|e| anyhow::anyhow!("Failed to get exec function: {}", e))?;
                exec_fn.call1((code_obj, namespace))
                    .map_err(|e| anyhow::anyhow!("Failed to execute optimizer code: {}", e))?;
                
                // Extract optimizer from namespace
                let optimizer = namespace.get_item("optimizer")
                    .map_err(|e| anyhow::anyhow!("Failed to get optimizer from namespace: {}", e))?;
                
                Ok(optimizer.to_object(py))
            })?;
            
            let mut optimizer_guard = self.optimizer.lock().await;
            if optimizer_guard.is_none() {
                *optimizer_guard = Some(optimizer_obj);
            }
        }
        
        // Perform training step and get results
        let (loss_val, grad_norm_val, clipped_grads_obj, should_update_optimizer) = Python::with_gil(|py| -> Result<(f64, f64, PyObject, bool)> {
            // Import MLX modules
            #[allow(deprecated)] // TODO: Migrate to PyModule::import_bound in future PyO3 version
            let mx = PyModule::import(py, "mlx.core")
                .map_err(|e| anyhow::anyhow!("Failed to import mlx.core: {}", e))?;
            #[allow(deprecated)] // TODO: Migrate to PyModule::import_bound in future PyO3 version
            let _nn = PyModule::import(py, "mlx.nn")
                .map_err(|e| anyhow::anyhow!("Failed to import mlx.nn: {}", e))?;
            
            // Convert input_ids to MLX array using Python code
            // This ensures correct dtype (uint32) for quantized models
            // First, find the maximum length to pad all sequences to the same length
            let max_len = input_ids.iter().map(|batch| batch.len()).max().unwrap_or(0);
            if max_len == 0 {
                return Err(anyhow::anyhow!("Empty input_ids batch"));
            }
            
            // Pad all sequences to max_len (using 0 as padding token)
            // Keep as u32 (unsigned) since token IDs are never negative
            let padded_input_ids: Vec<Vec<u32>> = input_ids.iter()
                .map(|batch| {
                    let mut padded = batch.iter().copied().collect::<Vec<u32>>();
                    padded.resize(max_len, 0); // Pad with 0
                    padded
                })
                .collect();
            
            // Use Python code to create MLX array with explicit dtype for quantized models
            // Build Python list structure manually for proper serialization
            let mut input_ids_py_str = String::from("[");
            for (i, batch) in padded_input_ids.iter().enumerate() {
                if i > 0 {
                    input_ids_py_str.push_str(", ");
                }
                input_ids_py_str.push('[');
                for (j, &token_id) in batch.iter().enumerate() {
                    if j > 0 {
                        input_ids_py_str.push_str(", ");
                    }
                    input_ids_py_str.push_str(&token_id.to_string());
                }
                input_ids_py_str.push(']');
            }
            input_ids_py_str.push(']');
            
            let create_array_code = format!(
                r#"
import mlx.core as mx
import numpy as np

# Convert input_ids to numpy array first (ensures correct dtype)
input_ids_np = np.array({}, dtype=np.uint32)
# Create MLX array from numpy array
# For quantized models, explicitly ensure uint32 dtype
input_ids_mlx = mx.array(input_ids_np)
# Ensure dtype is uint32 (cast if needed)
if input_ids_mlx.dtype != mx.uint32:
    input_ids_mlx = mx.astype(input_ids_mlx, mx.uint32)
"#,
                input_ids_py_str
            );
            
            // Compile and execute the array creation code
            #[allow(deprecated)] // TODO: Migrate to Python::import_bound in future PyO3 version
            let builtins = py.import("builtins")
                .map_err(|e| anyhow::anyhow!("Failed to import builtins: {}", e))?;
            let compile_fn = builtins.getattr("compile")
                .map_err(|e| anyhow::anyhow!("Failed to get compile function: {}", e))?;
            let code_obj = compile_fn.call1((create_array_code.as_str(), "<string>", "exec"))
                .map_err(|e| anyhow::anyhow!("Failed to compile array creation code: {}", e))?;
            
            #[allow(deprecated)] // TODO: Migrate to PyDict::new_bound in future PyO3 version
            let namespace = pyo3::types::PyDict::new(py);
            let exec_fn = builtins.getattr("exec")
                .map_err(|e| anyhow::anyhow!("Failed to get exec function: {}", e))?;
            exec_fn.call1((code_obj, namespace))
                .map_err(|e| anyhow::anyhow!("Failed to execute array creation code: {}", e))?;
            
            // Extract input_ids_mlx from namespace
            let input_ids_mlx = namespace.get_item("input_ids_mlx")
                .map_err(|e| anyhow::anyhow!("Failed to get input_ids_mlx from namespace: {}", e))?;
            let input_ids_mlx = input_ids_mlx.to_object(py);
            
            // Convert advantages to MLX array
            #[allow(deprecated)] // TODO: Migrate to PyList::empty_bound in future PyO3 version
            let advantages_list = PyList::empty(py);
            for &adv in advantages {
                advantages_list.append(adv)
                    .map_err(|e| anyhow::anyhow!("Failed to append advantage: {}", e))?;
            }
            let advantages_mlx = mx.call_method1("array", (advantages_list,))
                .map_err(|e| anyhow::anyhow!("Failed to create MLX array from advantages: {}", e))?;
            
            // Define loss function as Python code and execute it
            // This is simpler than trying to build it with PyO3 APIs
            let loss_fn_code = format!(
                r#"
def loss_fn(model, input_ids, advantages, kl_penalty_coeff, reward_weight):
    import mlx.core as mx
    import mlx.nn as nn
    
    # Ensure input_ids is uint32 for quantized models
    if input_ids.dtype != mx.uint32:
        input_ids = mx.astype(input_ids, mx.uint32)
    
    # Forward pass
    if hasattr(model, 'model'):
        logits = model.model(input_ids)
    elif hasattr(model, '__call__'):
        logits = model(input_ids)
    else:
        raise ValueError("Model does not support forward pass")
    
    # Compute log_probs: log_softmax(logits)
    log_probs = nn.log_softmax(logits, axis=-1)
    
    # Select log_probs for generated tokens (skip first token which is prompt)
    # Create a mask to ignore padding tokens (token_id == 0)
    batch_size, seq_len, vocab_size = log_probs.shape
    selected_log_probs = []
    mask = []
    
    for b in range(batch_size):
        batch_log_probs = []
        batch_mask = []
        for t in range(seq_len - 1):  # Skip first token
            token_id = int(input_ids[b, t + 1])
            # Skip padding tokens (token_id == 0)
            if token_id > 0 and token_id < vocab_size:
                batch_log_probs.append(float(log_probs[b, t, token_id]))
                batch_mask.append(1.0)
            else:
                # Padding token - add 0.0 but mark as masked
                batch_log_probs.append(0.0)
                batch_mask.append(0.0)
        selected_log_probs.append(batch_log_probs)
        mask.append(batch_mask)
    
    # Convert to MLX arrays
    if selected_log_probs and len(selected_log_probs[0]) > 0:
        # Pad all sequences to the same length (they should already be the same from input padding)
        max_len = max(len(b) for b in selected_log_probs)
        selected_log_probs_padded = []
        mask_padded = []
        for b in range(batch_size):
            log_probs_seq = selected_log_probs[b]
            mask_seq = mask[b]
            # Pad if needed (shouldn't be needed, but just in case)
            padded_log_probs = log_probs_seq + [0.0] * (max_len - len(log_probs_seq))
            padded_mask = mask_seq + [0.0] * (max_len - len(mask_seq))
            selected_log_probs_padded.append(padded_log_probs)
            mask_padded.append(padded_mask)
        
        selected_log_probs_array = mx.array(selected_log_probs_padded)
        mask_array = mx.array(mask_padded)
        
        # Expand advantages to match sequence length
        advantages_expanded = mx.reshape(advantages, (batch_size, 1))
        advantages_expanded = mx.broadcast_to(advantages_expanded, (batch_size, max_len))
        
        # Apply mask to ignore padding tokens
        masked_log_probs = selected_log_probs_array * mask_array
        masked_advantages = advantages_expanded * mask_array
        
        # Policy loss: -(log_probs * advantages).mean() over non-padded tokens
        # Sum over all tokens, then divide by number of non-padded tokens
        total_loss = -mx.sum(masked_log_probs * masked_advantages)
        num_valid_tokens = mx.sum(mask_array)
        if num_valid_tokens > 0:
            policy_loss = (total_loss / num_valid_tokens) * reward_weight
        else:
            policy_loss = mx.array(0.0)
    else:
        policy_loss = mx.array(0.0)
    
    # KL divergence (simplified - would need reference model)
    kl_penalty = mx.array(0.0)  # TODO: Compute actual KL divergence with ref_model
    
    # Total loss
    total_loss = policy_loss + kl_penalty_coeff * kl_penalty
    
    return total_loss
"#
            );
            
            // Compile and execute loss function using exec()
            #[allow(deprecated)] // TODO: Migrate to Python::import_bound in future PyO3 version
            let builtins = py.import("builtins")
                .map_err(|e| anyhow::anyhow!("Failed to import builtins: {}", e))?;
            let compile_fn = builtins.getattr("compile")
                .map_err(|e| anyhow::anyhow!("Failed to get compile function: {}", e))?;
            let code_obj = compile_fn.call1((loss_fn_code.as_str(), "<string>", "exec"))
                .map_err(|e| anyhow::anyhow!("Failed to compile loss function: {}", e))?;
            #[allow(deprecated)] // TODO: Migrate to PyDict::new_bound in future PyO3 version
            let namespace = PyDict::new(py);
            // Use exec() builtin function to execute the code object
            let exec_fn = builtins.getattr("exec")
                .map_err(|e| anyhow::anyhow!("Failed to get exec function: {}", e))?;
            exec_fn.call1((code_obj, namespace))
                .map_err(|e| anyhow::anyhow!("Failed to execute loss function code: {}", e))?;
            let loss_fn = namespace.get_item("loss_fn")
                .map_err(|e| anyhow::anyhow!("Failed to get loss_fn from namespace: {}", e))?;
            
            // Compute loss and gradients using MLX value_and_grad
            let value_and_grad = mx.getattr("value_and_grad")
                .map_err(|e| anyhow::anyhow!("Failed to get value_and_grad: {}", e))?;
            let loss_fn_wrapped = value_and_grad.call1((loss_fn,))
                .map_err(|e| anyhow::anyhow!("Failed to wrap loss function with value_and_grad: {}", e))?;
            
            let result = loss_fn_wrapped.call1((
                &self.model,
                input_ids_mlx,
                advantages_mlx,
                kl_penalty,
                reward_weight,
            ))
                .map_err(|e| anyhow::anyhow!("Failed to compute loss and gradients: {}", e))?;
            
            // Extract loss and gradients (result is a tuple: (loss, grads))
            let result_tuple = result.downcast::<PyTuple>()
                .map_err(|e| anyhow::anyhow!("Expected tuple (loss, grads), got: {}", e))?;
            let loss = result_tuple.get_item(0)
                .map_err(|e| anyhow::anyhow!("Failed to get loss from result: {}", e))?;
            let grads = result_tuple.get_item(1)
                .map_err(|e| anyhow::anyhow!("Failed to get grads from result: {}", e))?;

            // QLoRA: only update adapter/trainable params (do NOT touch quantized base weights)
            let grads = if self.use_lora {
                let filter_code = r#"
def _keep_trainable_grads(grads, trainable):
    # grads/tree can be nested dicts; trainable has same structure for trainable leaves
    if isinstance(trainable, dict) and isinstance(grads, dict):
        out = {}
        for k, tv in trainable.items():
            if k in grads:
                sub = _keep_trainable_grads(grads[k], tv)
                if sub is not None:
                    out[k] = sub
        return out if out else None
    # Leaf: keep gradient leaf for corresponding trainable leaf
    return grads
"#;
                #[allow(deprecated)] // TODO: Migrate to PyDict::new_bound in future PyO3 version
                let ns = PyDict::new(py);
                // compile/exec not necessary here, but use exec for consistency
                #[allow(deprecated)] // TODO: Migrate to Python::import_bound in future PyO3 version
                let builtins = py.import("builtins")
                    .map_err(|e| anyhow::anyhow!("Failed to import builtins: {}", e))?;
                let exec_fn = builtins.getattr("exec")
                    .map_err(|e| anyhow::anyhow!("Failed to get exec function: {}", e))?;
                exec_fn.call1((filter_code, ns))
                    .map_err(|e| anyhow::anyhow!("Failed to define trainable grad filter: {}", e))?;

                let keep_fn = ns
                    .get_item("_keep_trainable_grads")
                    .map_err(|e| anyhow::anyhow!("Failed to get _keep_trainable_grads: {}", e))?
                    .ok_or_else(|| anyhow::anyhow!("Python filter code did not define _keep_trainable_grads"))?;
                let trainable = self.model.bind(py).call_method0("trainable_parameters")
                    .map_err(|e| anyhow::anyhow!("Failed to call model.trainable_parameters(): {}", e))?;
                let filtered = keep_fn.call1((grads, trainable))
                    .map_err(|e| anyhow::anyhow!("Failed to filter grads to trainable params: {}", e))?;
                filtered
            } else {
                grads
            };
            
            // Scale loss for gradient accumulation
            let gradient_accumulation_steps_f64 = gradient_accumulation_steps as f64;
            let scaled_loss = mx.call_method1("divide", (loss, gradient_accumulation_steps_f64))
                .map_err(|e| anyhow::anyhow!("Failed to scale loss: {}", e))?;
            
            // Convert loss to float
            let loss_val = Self::mlx_to_float(py, scaled_loss)?;
            
            // Gradient clipping
            let grad_norm_sq = Self::compute_grad_norm_sq(py, mx, grads)?;
            let grad_norm = mx.call_method1("sqrt", (grad_norm_sq,))
                .map_err(|e| anyhow::anyhow!("Failed to compute grad_norm: {}", e))?;
            let grad_norm_val = Self::mlx_to_float(py, grad_norm)?;
            
            let clipped_grads = if grad_norm_val > max_grad_norm {
                let clip_factor = max_grad_norm / grad_norm_val;
                Self::clip_gradients(py, mx, grads, clip_factor)?
            } else {
                grads.to_object(py)
            };
            
            // Store clipped_grads for optimizer update (outside GIL)
            let clipped_grads_obj = clipped_grads;
            let should_update_optimizer = accumulation_step % gradient_accumulation_steps == 0;
            
            let elapsed = start_time.elapsed().as_secs_f64();
            debug!(
                "Training step completed in {:.3}s (loss: {:.4}, grad_norm: {:.4})",
                elapsed, loss_val, grad_norm_val
            );
            
            // Return results and flag for optimizer update
            Ok((loss_val, grad_norm_val, clipped_grads_obj, should_update_optimizer))
        })?;
        
        // Optimizer step (only on accumulation boundary) - done outside GIL to avoid blocking
        if should_update_optimizer {
            let optimizer_guard = self.optimizer.lock().await;
            if let Some(opt) = optimizer_guard.as_ref() {
                Python::with_gil(|py| -> Result<()> {
                    // MLX optimizer.update() takes (model, gradients) as separate arguments
                    // Use Python code execution for reliable argument passing
                    let update_code = r#"
optimizer.update(model, gradients)
"#;
                    
                    // Compile and execute the update code
                    #[allow(deprecated)] // TODO: Migrate to Python::import_bound in future PyO3 version
                    let builtins = py.import("builtins")
                        .map_err(|e| anyhow::anyhow!("Failed to import builtins: {}", e))?;
                    let compile_fn = builtins.getattr("compile")
                        .map_err(|e| anyhow::anyhow!("Failed to get compile function: {}", e))?;
                    let code_obj = compile_fn.call1((update_code, "<string>", "exec"))
                        .map_err(|e| anyhow::anyhow!("Failed to compile update code: {}", e))?;
                    
                    #[allow(deprecated)] // TODO: Migrate to PyDict::new_bound in future PyO3 version
                    let namespace = pyo3::types::PyDict::new(py);
                    namespace.set_item("optimizer", opt)
                        .map_err(|e| anyhow::anyhow!("Failed to set optimizer in namespace: {}", e))?;
                    namespace.set_item("model", &self.model)
                        .map_err(|e| anyhow::anyhow!("Failed to set model in namespace: {}", e))?;
                    namespace.set_item("gradients", &clipped_grads_obj)
                        .map_err(|e| anyhow::anyhow!("Failed to set gradients in namespace: {}", e))?;
                    
                    let exec_fn = builtins.getattr("exec")
                        .map_err(|e| anyhow::anyhow!("Failed to get exec function: {}", e))?;
                    exec_fn.call1((code_obj, namespace))
                        .map_err(|e| anyhow::anyhow!("Failed to update optimizer: {}", e))?;
                    
                    Ok(())
                })?;
            }
        }
        
        Ok((loss_val, grad_norm_val))
    }
    
    /// Convert MLX array to Rust f64
    fn mlx_to_float(py: Python, mlx_array: &PyAny) -> Result<f64> {
        // Try numpy conversion first
        #[allow(deprecated)] // TODO: Migrate to PyModule::import_bound in future PyO3 version
        let np = PyModule::import(py, "numpy")
            .map_err(|e| anyhow::anyhow!("Failed to import numpy: {}", e))?;
        let np_array = np.call_method1("array", (mlx_array,))
            .map_err(|e| anyhow::anyhow!("Failed to convert MLX array to numpy: {}", e))?;
        let item = np_array.call_method0("item")
            .map_err(|e| anyhow::anyhow!("Failed to get item from numpy array: {}", e))?;
        Ok(item.extract::<f64>()
            .map_err(|e| anyhow::anyhow!("Failed to extract f64 from item: {}", e))?)
    }
    
    /// Compute gradient norm squared
    fn compute_grad_norm_sq(py: Python, _mx: &PyModule, grads: &PyAny) -> Result<PyObject> {
        // Use Python code to compute gradient norm squared
        // MLX gradients can be nested dicts (for nested model structures)
        // We need to recursively flatten and compute norm
        let grad_norm_code = r#"
import mlx.core as mx

def compute_grad_norm_sq_recursive(grad_dict):
    """Recursively compute gradient norm squared, handling nested dicts"""
    total = mx.array(0.0)
    for grad in grad_dict.values():
        if isinstance(grad, dict):
            # Nested dict (e.g., for nested model structures)
            total = total + compute_grad_norm_sq_recursive(grad)
        elif hasattr(grad, 'shape'):
            # MLX array (has shape attribute)
            try:
                grad_sq = grad * grad
                grad_sum = mx.sum(grad_sq)
                total = total + grad_sum
            except (TypeError, AttributeError):
                # Skip if not a valid MLX array
                pass
        # Skip other types (None, lists, etc.)
    return total

# Compute gradient norm squared
grad_norm_sq = compute_grad_norm_sq_recursive(grads)
"#;
        
        // Compile and execute the code
        #[allow(deprecated)] // TODO: Migrate to Python::import_bound in future PyO3 version
        let builtins = py.import("builtins")
            .map_err(|e| anyhow::anyhow!("Failed to import builtins: {}", e))?;
        let compile_fn = builtins.getattr("compile")
            .map_err(|e| anyhow::anyhow!("Failed to get compile function: {}", e))?;
        let code_obj = compile_fn.call1((grad_norm_code, "<string>", "exec"))
            .map_err(|e| anyhow::anyhow!("Failed to compile grad_norm code: {}", e))?;
        
        #[allow(deprecated)] // TODO: Migrate to PyDict::new_bound in future PyO3 version
        let namespace = pyo3::types::PyDict::new(py);
        namespace.set_item("grads", grads)
            .map_err(|e| anyhow::anyhow!("Failed to set grads in namespace: {}", e))?;
        
        let exec_fn = builtins.getattr("exec")
            .map_err(|e| anyhow::anyhow!("Failed to get exec function: {}", e))?;
        exec_fn.call1((code_obj, namespace))
            .map_err(|e| anyhow::anyhow!("Failed to execute grad_norm code: {}", e))?;
        
        // Extract grad_norm_sq from namespace
        let grad_norm_sq = namespace.get_item("grad_norm_sq")
            .map_err(|e| anyhow::anyhow!("Failed to get grad_norm_sq from namespace: {}", e))?;
        
        Ok(grad_norm_sq.to_object(py))
    }
    
    /// Clip gradients by factor
    fn clip_gradients(
        py: Python,
        _mx: &PyModule,
        grads: &PyAny,
        clip_factor: f64,
    ) -> Result<PyObject> {
        // Use Python code to clip gradients
        // MLX gradients can be nested dicts (for nested model structures)
        // We need to recursively clip gradients
        let clip_code = format!(
            r#"
import mlx.core as mx

def clip_gradients_recursive(grad_dict, factor):
    """Recursively clip gradients, handling nested dicts"""
    clipped = {{}}
    for key, grad in grad_dict.items():
        if isinstance(grad, dict):
            # Nested dict (e.g., for nested model structures)
            clipped[key] = clip_gradients_recursive(grad, factor)
        else:
            # MLX array - clip by multiplying by factor
            clipped[key] = grad * factor
    return clipped

# Clip gradients
clipped = clip_gradients_recursive(grads, {})
"#,
            clip_factor
        );
        
        // Compile and execute the code
        #[allow(deprecated)] // TODO: Migrate to Python::import_bound in future PyO3 version
        let builtins = py.import("builtins")
            .map_err(|e| anyhow::anyhow!("Failed to import builtins: {}", e))?;
        let compile_fn = builtins.getattr("compile")
            .map_err(|e| anyhow::anyhow!("Failed to get compile function: {}", e))?;
        let code_obj = compile_fn.call1((clip_code.as_str(), "<string>", "exec"))
            .map_err(|e| anyhow::anyhow!("Failed to compile clip code: {}", e))?;
        
        #[allow(deprecated)] // TODO: Migrate to PyDict::new_bound in future PyO3 version
        let namespace = pyo3::types::PyDict::new(py);
        namespace.set_item("grads", grads)
            .map_err(|e| anyhow::anyhow!("Failed to set grads in namespace: {}", e))?;
        
        let exec_fn = builtins.getattr("exec")
            .map_err(|e| anyhow::anyhow!("Failed to get exec function: {}", e))?;
        exec_fn.call1((code_obj, namespace))
            .map_err(|e| anyhow::anyhow!("Failed to execute clip code: {}", e))?;
        
        // Extract clipped from namespace
        let clipped = namespace.get_item("clipped")
            .map_err(|e| anyhow::anyhow!("Failed to get clipped from namespace: {}", e))?;
        
        Ok(clipped.to_object(py))
    }
    
    /// Shutdown (no-op for PyO3, but kept for API compatibility)
    pub async fn shutdown(&self) -> Result<()> {
        info!("MLX trainer shutdown (model: {})", self.model_path);
        Ok(())
    }
}
