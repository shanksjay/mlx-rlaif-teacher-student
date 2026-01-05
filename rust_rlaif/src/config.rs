use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

// Nested YAML structure matching config.yaml
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelConfig {
    base_model: String,
    #[serde(default = "default_false")]
    use_4bit: bool,
    #[serde(default = "default_512")]
    max_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TeacherConfig {
    provider: String,
    model_name: String,
    #[serde(rename = "api_key_env")]
    api_key_env: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingConfig {
    output_dir: String,
    num_epochs: usize,
    batch_size: usize,
    gradient_accumulation_steps: usize,
    learning_rate: f64,
    warmup_steps: usize,
    save_steps: usize,
    eval_steps: usize,
    logging_steps: usize,
    max_grad_norm: f64,
    weight_decay: f64,
    lr_scheduler_type: String,
    #[serde(default = "default_one")]
    generation_accumulation_batches: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RlaifConfigSection {
    reward_weight: f64,
    kl_penalty: f64,
    #[serde(default = "default_false")]
    adaptive_kl_enabled: bool,
    #[serde(default = "default_0_075")]
    target_kl: f64,
    #[serde(default = "default_0_1")]
    kl_gain: f64,
    #[serde(default = "default_0_1")]
    beta: f64,
    num_samples_per_prompt: usize,
    #[serde(default = "default_one")]
    top_samples_per_prompt: usize,
    #[serde(default = "default_false")]
    use_advantage_normalization: bool,
    #[serde(default = "default_per_prompt")]
    advantage_baseline_type: String,
    #[serde(default = "default_0_8")]
    advantage_baseline_ema_alpha: f64,
    #[serde(default = "default_false")]
    use_tiered_scoring: bool,
    #[serde(default = "default_0_3")]
    heuristic_score_threshold: f64,
    #[serde(default = "default_false")]
    truncate_prompt_for_scoring: bool,
    #[serde(default = "default_200")]
    prompt_context_chars: usize,
    #[serde(default = "default_false")]
    move_rubric_to_system_prompt: bool,
    #[serde(default = "default_false")]
    use_frozen_reference_for_kl: bool,
    #[serde(default = "default_false")]
    curriculum_learning: bool,
    #[serde(default = "default_one")]
    generation_accumulation_batches: usize,
    #[serde(default = "default_0_7")]
    generation_temperature: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DataConfig {
    train_file: String,
    eval_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LoggingConfig {
    #[serde(default = "default_tensorboard_dir")]
    tensorboard_dir: String,
    #[serde(default = "default_info")]
    log_level: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            tensorboard_dir: default_tensorboard_dir(),
            log_level: default_info(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HardwareConfig {
    #[serde(default = "default_true")]
    use_mps: bool,
    #[serde(default = "default_bf16")]
    mixed_precision: String,
    #[serde(default = "default_false")]
    use_mlx_for_generation: bool,
    #[serde(default = "default_false")]
    save_mlx_format: bool,
    #[serde(default)]
    mlx_quantization: Option<String>,
    #[serde(default)]
    mlx_model_path: Option<String>,  // MLX model for fast forward passes (generation)
    #[serde(default)]
    coreml_model_path: Option<String>,  // CoreML model for backward passes (training)
    #[serde(default = "default_false")]
    use_mlx_generation_worker: bool,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            use_mps: default_true(),
            mixed_precision: default_bf16(),
            use_mlx_for_generation: default_false(),
            save_mlx_format: default_false(),
            mlx_quantization: None,
            mlx_model_path: None,
            coreml_model_path: None,
            use_mlx_generation_worker: default_false(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct YamlConfig {
    model: ModelConfig,
    teacher: TeacherConfig,
    training: TrainingConfig,
    rlaif: RlaifConfigSection,
    data: DataConfig,
    #[serde(default)]
    logging: LoggingConfig,
    #[serde(default)]
    hardware: HardwareConfig,
}

// Helper functions for defaults
fn default_false() -> bool { false }
fn default_true() -> bool { true }
fn default_one() -> usize { 1 }
fn default_512() -> usize { 512 }
fn default_0_075() -> f64 { 0.075 }
fn default_0_1() -> f64 { 0.1 }
fn default_0_3() -> f64 { 0.3 }
fn default_0_7() -> f64 { 0.7 }
fn default_0_8() -> f64 { 0.8 }
fn default_200() -> usize { 200 }
fn default_per_prompt() -> String { "per_prompt".to_string() }
fn default_tensorboard_dir() -> String { "./logs/tensorboard".to_string() }
fn default_info() -> String { "info".to_string() }
fn default_bf16() -> String { "bf16".to_string() }

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RlaifConfig {
    // Model configuration
    pub base_model: String,
    pub use_4bit: bool,
    pub use_mps: bool,
    pub mixed_precision: String,
    pub max_length: usize,

    // Teacher configuration
    pub teacher_provider: String,
    pub teacher_model: String,
    pub teacher_api_key_env: String,

    // Training configuration
    pub num_epochs: usize,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub save_steps: usize,
    pub eval_steps: usize,
    pub logging_steps: usize,
    pub max_grad_norm: f64,
    pub weight_decay: f64,
    pub lr_scheduler_type: String,

    // RLAIF specific
    pub reward_weight: f64,
    pub kl_penalty: f64,
    pub beta: f64,
    pub num_samples_per_prompt: usize,

    // Data configuration
    pub train_file: String,
    pub eval_file: Option<String>,

    // Output configuration
    pub output_dir: String,
    pub tensorboard_dir: String,
    pub log_level: String,

    // Advanced features
    pub adaptive_kl_enabled: bool,
    pub target_kl: f64,
    pub kl_gain: f64,
    pub top_samples_per_prompt: usize,
    pub use_advantage_normalization: bool,
    pub advantage_baseline_type: String,
    pub advantage_baseline_ema_alpha: f64,
    pub use_tiered_scoring: bool,
    pub heuristic_score_threshold: f64,
    pub truncate_prompt_for_scoring: bool,
    pub prompt_context_chars: usize,
    pub move_rubric_to_system_prompt: bool,
    pub use_frozen_reference_for_kl: bool,
    pub generation_accumulation_batches: usize,
    pub use_mlx_for_generation: bool,
    pub save_mlx_format: bool,
    pub mlx_quantization: Option<String>,
    pub mlx_model_path: Option<String>,  // MLX model for fast forward passes (generation)
    pub coreml_model_path: Option<String>,  // CoreML model for backward passes (training)
    pub use_mlx_generation_worker: bool,
    pub generation_temperature: Option<f64>,
    pub curriculum_learning: bool,
    pub curriculum_mix_difficulty: bool,
    pub curriculum_num_buckets: usize,
}

impl Default for RlaifConfig {
    fn default() -> Self {
        Self {
            base_model: "Qwen/Qwen2.5-Coder-3B-Instruct".to_string(),
            use_4bit: false,
            use_mps: true,
            mixed_precision: "bf16".to_string(),
            max_length: 2048,
            teacher_provider: "anthropic".to_string(),
            teacher_model: "claude-3-5-haiku-20241022".to_string(),
            teacher_api_key_env: "ANTHROPIC_API_KEY".to_string(),
            num_epochs: 3,
            batch_size: 4,
            gradient_accumulation_steps: 1,
            learning_rate: 1.5e-5,
            warmup_steps: 100,
            save_steps: 500,
            eval_steps: 500,
            logging_steps: 3,
            max_grad_norm: 1.0,
            weight_decay: 0.01,
            lr_scheduler_type: "cosine".to_string(),
            reward_weight: 1.0,
            kl_penalty: 0.1,
            beta: 0.1,
            num_samples_per_prompt: 4,
            train_file: "./data/train.jsonl".to_string(),
            eval_file: Some("./data/eval.jsonl".to_string()),
            output_dir: "./checkpoints".to_string(),
            tensorboard_dir: "./logs/tensorboard".to_string(),
            log_level: "info".to_string(),
            adaptive_kl_enabled: true,
            target_kl: 0.075,
            kl_gain: 0.1,
            top_samples_per_prompt: 1,
            use_advantage_normalization: true,
            advantage_baseline_type: "per_prompt".to_string(),
            advantage_baseline_ema_alpha: 0.9,
            use_tiered_scoring: true,
            heuristic_score_threshold: 0.3,
            truncate_prompt_for_scoring: true,
            prompt_context_chars: 200,
            move_rubric_to_system_prompt: true,
            use_frozen_reference_for_kl: true,
            generation_accumulation_batches: 1,
            use_mlx_for_generation: false,
            save_mlx_format: false,
            mlx_quantization: None,
            mlx_model_path: None,
            coreml_model_path: None,
            use_mlx_generation_worker: false,
            generation_temperature: Some(0.7),
            curriculum_learning: false,
            curriculum_mix_difficulty: true,
            curriculum_num_buckets: 8,
        }
    }
}

impl RlaifConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config file: {:?}", path.as_ref()))?;
        
        let yaml_config: YamlConfig = serde_yaml::from_str(&content)
            .context("Failed to parse YAML config")?;
        
        // Convert nested YAML structure to flat RlaifConfig
        Ok(Self {
            base_model: yaml_config.model.base_model,
            use_4bit: yaml_config.model.use_4bit,
            use_mps: yaml_config.hardware.use_mps,
            mixed_precision: yaml_config.hardware.mixed_precision,
            max_length: yaml_config.model.max_length,
            teacher_provider: yaml_config.teacher.provider,
            teacher_model: yaml_config.teacher.model_name,
            teacher_api_key_env: yaml_config.teacher.api_key_env,
            num_epochs: yaml_config.training.num_epochs,
            batch_size: yaml_config.training.batch_size,
            gradient_accumulation_steps: yaml_config.training.gradient_accumulation_steps,
            learning_rate: yaml_config.training.learning_rate,
            warmup_steps: yaml_config.training.warmup_steps,
            save_steps: yaml_config.training.save_steps,
            eval_steps: yaml_config.training.eval_steps,
            logging_steps: yaml_config.training.logging_steps,
            max_grad_norm: yaml_config.training.max_grad_norm,
            weight_decay: yaml_config.training.weight_decay,
            lr_scheduler_type: yaml_config.training.lr_scheduler_type,
            reward_weight: yaml_config.rlaif.reward_weight,
            kl_penalty: yaml_config.rlaif.kl_penalty,
            beta: yaml_config.rlaif.beta,
            num_samples_per_prompt: yaml_config.rlaif.num_samples_per_prompt,
            train_file: yaml_config.data.train_file,
            eval_file: yaml_config.data.eval_file,
            output_dir: yaml_config.training.output_dir,
            tensorboard_dir: yaml_config.logging.tensorboard_dir,
            log_level: yaml_config.logging.log_level,
            adaptive_kl_enabled: yaml_config.rlaif.adaptive_kl_enabled,
            target_kl: yaml_config.rlaif.target_kl,
            kl_gain: yaml_config.rlaif.kl_gain,
            top_samples_per_prompt: yaml_config.rlaif.top_samples_per_prompt,
            use_advantage_normalization: yaml_config.rlaif.use_advantage_normalization,
            advantage_baseline_type: yaml_config.rlaif.advantage_baseline_type,
            advantage_baseline_ema_alpha: yaml_config.rlaif.advantage_baseline_ema_alpha,
            use_tiered_scoring: yaml_config.rlaif.use_tiered_scoring,
            heuristic_score_threshold: yaml_config.rlaif.heuristic_score_threshold,
            truncate_prompt_for_scoring: yaml_config.rlaif.truncate_prompt_for_scoring,
            prompt_context_chars: yaml_config.rlaif.prompt_context_chars,
            move_rubric_to_system_prompt: yaml_config.rlaif.move_rubric_to_system_prompt,
            use_frozen_reference_for_kl: yaml_config.rlaif.use_frozen_reference_for_kl,
            generation_accumulation_batches: yaml_config.rlaif.generation_accumulation_batches,
            use_mlx_for_generation: yaml_config.hardware.use_mlx_for_generation,
            save_mlx_format: yaml_config.hardware.save_mlx_format,
            mlx_quantization: yaml_config.hardware.mlx_quantization,
            mlx_model_path: yaml_config.hardware.mlx_model_path,
            coreml_model_path: yaml_config.hardware.coreml_model_path.clone(),
            use_mlx_generation_worker: yaml_config.hardware.use_mlx_generation_worker,
            generation_temperature: Some(yaml_config.rlaif.generation_temperature),
            curriculum_learning: yaml_config.rlaif.curriculum_learning,
            curriculum_mix_difficulty: true, // Default value
            curriculum_num_buckets: 8, // Default value
        })
    }

    #[allow(dead_code)]
    pub fn save<P: AsRef<Path>>(&self, _path: P) -> Result<()> {
        // Note: Saving would require converting back to YamlConfig structure
        // For now, this is a placeholder
        anyhow::bail!("Config saving not yet implemented")
    }
}

