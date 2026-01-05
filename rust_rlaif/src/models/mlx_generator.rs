use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};
use std::path::Path;
use tracing::{debug, error, info, warn};

/// MLX Generator for fast forward passes (generation)
/// Uses PyO3 to call MLX directly from Rust (no subprocess overhead)
pub struct MlxGenerator {
    model: PyObject,
    tokenizer: PyObject,
    model_path: String,
}

impl MlxGenerator {
    /// Load MLX model and tokenizer for generation
    pub async fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let model_path_str = model_path.as_ref().to_string_lossy().to_string();
        info!("Loading MLX model for generation: {}", model_path_str);

        // Initialize Python if not already initialized
        pyo3::prepare_freethreaded_python();
        
        // Acquire Python GIL and load model
        let (model, tokenizer) = Python::with_gil(|py| -> Result<(PyObject, PyObject)> {
            // Get current Python executable for error messages
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
            let result = load_fn.call1(args)
                .map_err(|e| anyhow::anyhow!("Failed to call mlx_lm.load: {}", e))?;
            
            // Extract model and tokenizer (both elements of tuple)
            let model_tuple = result.downcast::<PyTuple>()
                .map_err(|e| anyhow::anyhow!("mlx_lm.load should return a tuple (model, tokenizer), got: {}", e))?;
            let model = model_tuple.get_item(0)
                .map_err(|e| anyhow::anyhow!("Failed to get model from load result: {}", e))?;
            let tokenizer = model_tuple.get_item(1)
                .map_err(|e| anyhow::anyhow!("Failed to get tokenizer from load result: {}", e))?;
            
            // Get the underlying model (mlx_lm wraps it in a Model class)
            let model_obj = if model.hasattr("model")? {
                model.getattr("model")
                    .map_err(|e| anyhow::anyhow!("Failed to get model.model attribute: {}", e))?
                    .to_object(py)
            } else {
                model.to_object(py)
            };
            
            info!("MLX model and tokenizer loaded successfully");
            Ok((model_obj, tokenizer.to_object(py)))
        })?;
        
        Ok(Self {
            model,
            tokenizer,
            model_path: model_path_str,
        })
    }
    
    /// Generate text using MLX model (fast forward pass)
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        top_k: usize,
    ) -> Result<String> {
        Python::with_gil(|py| -> Result<String> {
            // Call mlx_lm.generate using Python code for flexibility
            // This handles sampler creation and function call in one go
            let generate_code = if temperature > 0.0 {
                format!(
                    r#"
import mlx_lm
try:
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp={}, top_p={}, top_k={})
except Exception:
    sampler = None

result = mlx_lm.generate(model, tokenizer, prompt={:?}, max_tokens={}, sampler=sampler)
"#,
                    temperature, top_p, top_k, prompt, max_tokens
                )
            } else {
                format!(
                    r#"
import mlx_lm
result = mlx_lm.generate(model, tokenizer, prompt={:?}, max_tokens={})
"#,
                    prompt, max_tokens
                )
            };
            
            // Compile and execute the generation code using exec()
            #[allow(deprecated)] // TODO: Migrate to Python::import_bound in future PyO3 version
            let builtins = py.import("builtins")
                .map_err(|e| anyhow::anyhow!("Failed to import builtins: {}", e))?;
            let compile_fn = builtins.getattr("compile")
                .map_err(|e| anyhow::anyhow!("Failed to get compile function: {}", e))?;
            let code_obj = compile_fn.call1((generate_code.as_str(), "<string>", "exec"))
                .map_err(|e| anyhow::anyhow!("Failed to compile generation code: {}", e))?;
            
            #[allow(deprecated)] // TODO: Migrate to PyDict::new_bound in future PyO3 version
            let namespace = pyo3::types::PyDict::new(py);
            namespace.set_item("model", &self.model)
                .map_err(|e| anyhow::anyhow!("Failed to set model in namespace: {}", e))?;
            namespace.set_item("tokenizer", &self.tokenizer)
                .map_err(|e| anyhow::anyhow!("Failed to set tokenizer in namespace: {}", e))?;
            
            // Use exec() builtin function to execute the code object
            let exec_fn = builtins.getattr("exec")
                .map_err(|e| anyhow::anyhow!("Failed to get exec function: {}", e))?;
            exec_fn.call1((code_obj, namespace))
                .map_err(|e| anyhow::anyhow!("Failed to execute generation code: {}", e))?;
            
            // Extract generated text from namespace
            let result = namespace.get_item("result")
                .map_err(|e| anyhow::anyhow!("Failed to get result from namespace: {}", e))?
                .ok_or_else(|| anyhow::anyhow!("Generation code did not set 'result' variable"))?;
            
            let generated_text = result.extract::<String>()
                .map_err(|e| anyhow::anyhow!("Failed to extract generated text: {}", e))?;
            
            debug!("Generated {} characters", generated_text.len());
            Ok(generated_text)
        })
    }
    
    /// Shutdown (no-op for PyO3, but kept for API compatibility)
    pub async fn shutdown(&self) -> Result<()> {
        info!("MLX generator shutdown (model: {})", self.model_path);
        Ok(())
    }
}
