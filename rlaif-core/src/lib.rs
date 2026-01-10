use pyo3::prelude::*;
use sys_info;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct SystemInfo {
    total_ram_kb: u64,
    free_ram_kb: u64,
    cpu_cores: u32,
    os_type: String,
    gpu_available: bool,
}

/// Detects system capabilities and returns a JSON string
#[pyfunction]
fn detect_system_capabilities() -> PyResult<String> {
    let mem = sys_info::mem_info().unwrap_or(sys_info::MemInfo{ total: 0, free: 0, avail: 0, buffers: 0, cached: 0, swap_total: 0, swap_free: 0 });
    let cpu_num = sys_info::cpu_num().unwrap_or(0);
    let os_type = sys_info::os_type().unwrap_or("Unknown".to_string());

    // Very basic check for nvidia-smi as a proxy for GPU presence on Linux
    let gpu_available = std::process::Command::new("nvidia-smi")
        .output()
        .is_ok();

    let info = SystemInfo {
        total_ram_kb: mem.total,
        free_ram_kb: mem.free,
        cpu_cores: cpu_num,
        os_type,
        gpu_available,
    };

    serde_json::to_string(&info).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[derive(Serialize, Deserialize)]
struct AgenticFlowDefinition {
    task_intent: String,
    agent_behavior: String,
    constraints: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct UIConfiguration {
    inputs: Vec<String>,
    outputs: Vec<String>,
}

#[pyfunction]
fn validate_agent_config(json_config: String) -> PyResult<bool> {
    let _config: AgenticFlowDefinition = serde_json::from_str(&json_config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(true)
}

#[pyfunction]
fn validate_ui_config(json_config: String) -> PyResult<bool> {
    let _config: UIConfiguration = serde_json::from_str(&json_config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(true)
}

/// Core Training Loop implemented in Rust
#[pyclass]
struct TrainingLoop {
    step_count: usize,
}

#[pymethods]
impl TrainingLoop {
    #[new]
    fn new() -> Self {
        TrainingLoop { step_count: 0 }
    }

    /// Run a training step by calling back into Python backend
    fn run_step(&mut self, backend: PyObject, prompts: Vec<String>, config: String) -> PyResult<String> {
        self.step_count += 1;

        Python::with_gil(|py| {
            // 1. Call generate_samples on Python backend
            // Note: In PyO3, we call methods on the PyObject.
            // We assume backend has `generate_samples(prompts, config)`
            let config_dict = py.eval("dict()", None, None)?; // Passing empty dict for now, or parse JSON
            // For simplicity, we just pass the raw config string if the python side handles it,
            // but let's assume the python side expects a dict.
            // Let's just pass the config string for now to keep it simple or use serde to convert.
            // Actually, let's pass the config as a string and let Python parse it if needed, or pass a dict.
            // Backend signature: generate_samples(self, prompts: List[str], config: Dict[str, Any])

            // Just creating a dummy dict for config
            let py_config = py.eval("{}", None, None)?;

            let samples_obj = backend.call_method1(py, "generate_samples", (prompts, py_config))?;
            let samples: Vec<String> = samples_obj.extract(py)?;

            // 2. Score samples (mock logic in Rust or call another Python module)
            // For now, let's say Rust calculates a dummy score or calls a scorer
            // Let's pretend we have scores.
            let scores: Vec<f64> = vec![0.95; samples.len()]; // Mock scores

            // 3. Update weights
            // backend.update_weights(loss)
            let loss = 0.42; // Dummy loss
            backend.call_method1(py, "update_weights", (loss,))?;

            // Return a summary JSON
            let summary = format!(
                "{{\"step\": {}, \"samples_count\": {}, \"loss\": {}}}",
                self.step_count, samples.len(), loss
            );
            Ok(summary)
        })
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn rlaif_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_system_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(validate_agent_config, m)?)?;
    m.add_function(wrap_pyfunction!(validate_ui_config, m)?)?;
    m.add_class::<TrainingLoop>()?;
    Ok(())
}
