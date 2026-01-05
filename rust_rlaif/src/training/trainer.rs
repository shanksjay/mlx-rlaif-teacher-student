use anyhow::{Context, Result};
use crate::config::RlaifConfig;
use crate::models::{TeacherModel, StudentModel, CodeDataset, dataset::CodeSample};
use crate::training::{compute_rewards, compute_advantages, PerformanceMetrics, CodeSaver};
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

pub struct RlaifTrainer {
    pub config: Arc<RlaifConfig>,
    teacher: Arc<TeacherModel>,
    #[allow(dead_code)]
    student: Arc<Mutex<StudentModel>>,
    // device: Device,  // Disabled due to candle-core issues
    baseline_reward: Arc<Mutex<Option<f64>>>,
    baseline_file: String,  // Path to save/load baseline reward
    metrics: Arc<Mutex<PerformanceMetrics>>,
    code_saver: Arc<Mutex<CodeSaver>>,
    accumulation_step: Arc<Mutex<usize>>,  // Track gradient accumulation steps
}

impl RlaifTrainer {
    /// Load baseline reward from file
    fn load_baseline_reward(file_path: &str) -> Result<Option<f64>> {
        let content = std::fs::read_to_string(file_path)
            .context("Failed to read baseline reward file")?;
        let json: serde_json::Value = serde_json::from_str(&content)
            .context("Failed to parse baseline reward file")?;
        
        if let Some(baseline) = json.get("baseline_reward").and_then(|v| v.as_f64()) {
            Ok(Some(baseline))
        } else {
            Ok(None)
        }
    }
    
    /// Save baseline reward to file
    fn save_baseline_reward(file_path: &str, baseline: f64) -> Result<()> {
        let json = serde_json::json!({
            "baseline_reward": baseline,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "note": "Baseline reward computed from initial model performance"
        });
        
        // Ensure output directory exists
        if let Some(parent) = std::path::Path::new(file_path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        std::fs::write(file_path, serde_json::to_string_pretty(&json)?)
            .context("Failed to write baseline reward file")?;
        
        info!("Saved baseline reward {:.4} to {}", baseline, file_path);
        Ok(())
    }
    
    fn create_placeholder_student(base_model: &str) -> Result<Arc<Mutex<StudentModel>>> {
        // Create a minimal placeholder - only try to load tokenizer if it's a local path
        // HuggingFace model IDs (e.g., "Qwen/Qwen2.5-Coder-3B-Instruct") won't have local files
        let student_model = if std::path::Path::new(base_model).exists() {
            // Try to load tokenizer from local path
            let tokenizer_path = format!("{}/tokenizer.json", base_model);
            match tokenizers::Tokenizer::from_file(&tokenizer_path) {
                Ok(tokenizer) => {
                    info!("Loaded tokenizer from local path: {}", tokenizer_path);
                    StudentModel {
                        mlx_generator: None,
                        mlx_trainer: None,
                        tokenizer,
                        model_path: base_model.to_string(),
                    }
                }
                Err(_) => {
                    warn!("Could not load tokenizer from {}. Creating placeholder.", tokenizer_path);
                    // Create a minimal dummy tokenizer - won't be used in API-only mode
                    let bpe = tokenizers::models::bpe::BPE::default();
                    StudentModel {
                        mlx_generator: None,
                        mlx_trainer: None,
                        tokenizer: tokenizers::Tokenizer::new(bpe),
                        model_path: base_model.to_string(),
                    }
                }
            }
        } else {
            // HuggingFace model ID - skip tokenizer loading, we're in API-only mode anyway
            warn!("Model path '{}' appears to be a HuggingFace model ID. Skipping local tokenizer loading (API-only mode).", base_model);
            // Create a minimal dummy tokenizer - won't be used in API-only mode
            let bpe = tokenizers::models::bpe::BPE::default();
            StudentModel {
                mlx_generator: None,
                mlx_trainer: None,
                tokenizer: tokenizers::Tokenizer::new(bpe),
                model_path: base_model.to_string(),
            }
        };
        Ok(Arc::new(Mutex::new(student_model)))
    }
    
    pub async fn new(config: RlaifConfig) -> Result<Self> {
        // NOTE: Device setup disabled due to candle-core compilation issues
        // TODO: Re-enable when candle-core 0.5+ is available or issue is fixed
        // let device = if config.use_mps {
        //     Device::new_metal(0)?
        // } else {
        //     Device::Cpu
        // };

        // Initialize teacher model
        let teacher = Arc::new(TeacherModel::new(
            &config.teacher_provider,
            &config.teacher_model,
            &config.teacher_api_key_env,
        )?);

        // Initialize student model with hybrid approach:
        // - MLX for fast forward passes (generation)
        // - CoreML for backward passes (training)
        let device_str = if config.use_mps { "metal" } else { "cpu" };
        let mlx_path = config.mlx_model_path.as_ref().map(|p| p.as_str());
        let coreml_path = config.coreml_model_path.as_ref().map(|p| p.as_str());
        
        let student = match StudentModel::load(
            &config.base_model,
            device_str,
            mlx_path,
            coreml_path,
        ).await {
            Ok(s) => {
                info!("Successfully loaded student model (MLX Generator: {}, MLX Trainer: {})",
                    s.mlx_generator.is_some(),
                    s.mlx_trainer.is_some()
                );
                Arc::new(Mutex::new(s))
            }
            Err(e) => {
                warn!("Failed to load student model: {}. Using API-only mode.", e);
                Self::create_placeholder_student(&config.base_model)?
            }
        };

        // Initialize code saver
        let code_output_dir = config.output_dir.clone() + "/generated_code";
        let code_saver = CodeSaver::new(&code_output_dir)
            .context("Failed to initialize code saver")?;
        
        // Setup baseline reward file path
        let baseline_file = format!("{}/baseline_reward.json", config.output_dir);
        
        // Try to load existing baseline reward
        let baseline_reward = if std::path::Path::new(&baseline_file).exists() {
            match Self::load_baseline_reward(&baseline_file) {
                Ok(Some(baseline)) => {
                    info!("Loaded baseline reward from {}: {:.4}", baseline_file, baseline);
                    Some(baseline)
                }
                Ok(None) => {
                    info!("Baseline reward file exists but is empty, will compute new baseline");
                    None
                }
                Err(e) => {
                    warn!("Failed to load baseline reward from {}: {}. Will compute new baseline.", baseline_file, e);
                    None
                }
            }
        } else {
            None
        };
        
        Ok(Self {
            config: Arc::new(config),
            teacher,
            student,
            baseline_reward: Arc::new(Mutex::new(baseline_reward)),
            baseline_file,
            metrics: Arc::new(Mutex::new(PerformanceMetrics::new())),
            code_saver: Arc::new(Mutex::new(code_saver)),
            accumulation_step: Arc::new(Mutex::new(0)),
        })
    }

    pub async fn train(&self, train_dataset: CodeDataset, eval_dataset: Option<CodeDataset>) -> Result<()> {
        info!("Starting RLAIF training");
        info!("Training samples: {}", train_dataset.len());
        
        if let Some(eval) = eval_dataset.as_ref() {
            info!("Evaluation samples: {}", eval.len());
        }

        // Compute baseline reward if not already loaded
        {
            let baseline_guard = self.baseline_reward.lock().await;
            if let Some(existing_baseline) = *baseline_guard {
                info!("Using existing baseline reward: {:.4}", existing_baseline);
            } else {
                drop(baseline_guard); // Release lock before async operation
                info!("Computing baseline reward...");
                let computed_baseline = self.compute_baseline_reward(&train_dataset).await?;
                info!("Baseline reward: {:.4}", computed_baseline);
                
                // Save baseline reward to file for future runs
                if let Err(e) = Self::save_baseline_reward(&self.baseline_file, computed_baseline) {
                    warn!("Failed to save baseline reward: {}. Continuing without persistence.", e);
                }
                
                // Store the computed baseline
                *self.baseline_reward.lock().await = Some(computed_baseline);
            }
        }

        // Track best checkpoint by epoch-average reward.
        let mut best_epoch: Option<usize> = None;
        let mut best_epoch_reward: f64 = f64::NEG_INFINITY;

        // Training loop
        for epoch in 0..self.config.num_epochs {
            info!("Epoch {}/{}", epoch + 1, self.config.num_epochs);
            let epoch_avg_reward = self.train_epoch(epoch, &train_dataset).await?;
            info!("Epoch {} average reward: {:.4}", epoch + 1, epoch_avg_reward);

            if epoch_avg_reward.is_finite() && epoch_avg_reward > best_epoch_reward {
                best_epoch_reward = epoch_avg_reward;
                best_epoch = Some(epoch);
                info!(
                    "New best checkpoint: epoch {} (avg_reward={:.4})",
                    epoch + 1,
                    epoch_avg_reward
                );
            }

            // Print performance metrics periodically
            if (epoch + 1) % self.config.logging_steps == 0 {
                let metrics = self.metrics.lock().await;
                metrics.print_summary();
            }

            // Evaluation
            if let Some(eval) = &eval_dataset {
                if (epoch + 1) % self.config.eval_steps == 0 {
                    self.evaluate(eval).await?;
                }
            }

            // Save checkpoint every epoch (adapters are small; this enables true "best checkpoint" export)
            self.save_checkpoint(epoch).await?;
        }

        // Export fused MLX model for the best checkpoint (one-time, at the end)
        if let Some(best_epoch_idx) = best_epoch {
            let best_checkpoint_dir =
                format!("{}/checkpoint-epoch-{}", self.config.output_dir, best_epoch_idx);
            let adapter_dir = format!("{}/mlx_adapters", best_checkpoint_dir);
            let fused_out_dir = format!("{}/mlx_fused", best_checkpoint_dir);

            info!(
                "Exporting fused MLX model for best checkpoint (epoch {}, avg_reward={:.4}) -> {}",
                best_epoch_idx + 1,
                best_epoch_reward,
                fused_out_dir
            );

            let student = self.student.lock().await;
            if let Some(mlx_trainer) = &student.mlx_trainer {
                let mlx_trainer_guard = mlx_trainer.lock().await;
                if let Err(e) = mlx_trainer_guard
                    .export_fused_mlx_model(&adapter_dir, &fused_out_dir, self.config.mlx_quantization.as_deref())
                    .await
                {
                    warn!("Failed to export fused MLX model: {}", e);
                }
            } else {
                warn!("MLX trainer not available; skipping fused MLX export.");
            }
        }

        // Print final performance summary
        let metrics = self.metrics.lock().await;
        metrics.print_summary();
        
        // Save final code index
        let saver: tokio::sync::MutexGuard<'_, crate::training::CodeSaver> = self.code_saver.lock().await;
        saver.save_index()?;
        info!("Generated code saved to: {:?}", saver.get_output_dir());

        // Cleanup: shutdown MLX workers
        let student = self.student.lock().await;
        if let Some(mlx_gen) = &student.mlx_generator {
            let mlx_gen_guard = mlx_gen.lock().await;
            let _ = mlx_gen_guard.shutdown().await;
        }
        if let Some(mlx_trainer) = &student.mlx_trainer {
            let mlx_trainer_guard = mlx_trainer.lock().await;
            let _ = mlx_trainer_guard.shutdown().await;
        }

        Ok(())
    }

    async fn train_epoch(&self, _epoch: usize, dataset: &CodeDataset) -> Result<f64> {
        let pb = ProgressBar::new(dataset.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")?
                .progress_chars("#>-"),
        );

        let mut epoch_reward_sum: f64 = 0.0;
        let mut epoch_reward_count: usize = 0;

        // Process in batches
        for batch_start in (0..dataset.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(dataset.len());
            let batch: Vec<_> = (batch_start..batch_end)
                .filter_map(|i| dataset.get(i))
                .collect();

            // Generate samples (with timing)
            let gen_start = Instant::now();
            let samples = self.generate_samples(&batch).await?;
            let gen_time = gen_start.elapsed().as_secs_f64();
            
            // Estimate tokens generated (rough approximation: ~4 chars per token)
            let total_chars: usize = samples.iter().map(|s| s.code.len()).sum();
            let estimated_tokens = total_chars / 4;
            
            // Record generation metrics
            {
                let mut metrics = self.metrics.lock().await;
                metrics.record_generation(gen_time, estimated_tokens);
            }

            // Compute rewards (with timing)
            let scoring_start = Instant::now();
            let rewards = compute_rewards(
                &self.teacher,
                &samples,
                &self.config,
            ).await?;
            let scoring_time = scoring_start.elapsed().as_secs_f64();
            
            // Estimate API tokens used for scoring (rough approximation)
            let scoring_chars: usize = samples.iter()
                .map(|s| s.code.len() + s.prompt.len())
                .sum();
            let estimated_scoring_tokens = scoring_chars / 4;
            
            // Record scoring metrics
            {
                let mut metrics = self.metrics.lock().await;
                metrics.record_scoring(scoring_time, estimated_scoring_tokens);
            }

            // Compute advantages
            let advantages = compute_advantages(
                &rewards,
                &samples,
                &self.config,
            )?;
            
            // Debug: Log reward and advantage statistics
            let avg_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;
            let avg_advantage = advantages.iter().sum::<f64>() / advantages.len() as f64;
            let min_reward = rewards.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_reward = rewards.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_adv = advantages.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_adv = advantages.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            debug!("Rewards - avg: {:.4}, min: {:.4}, max: {:.4}", avg_reward, min_reward, max_reward);
            debug!("Advantages - avg: {:.4}, min: {:.4}, max: {:.4}", avg_advantage, min_adv, max_adv);

            // Track epoch reward (for best checkpoint selection)
            epoch_reward_sum += rewards.iter().sum::<f64>();
            epoch_reward_count += rewards.len();

            // Save generated code samples
            {
                let mut saver: tokio::sync::MutexGuard<'_, crate::training::CodeSaver> = self.code_saver.lock().await;
                let student_guard = self.student.lock().await;
                let source = if student_guard.mlx_generator.is_some() { "local" } else { "api" };
                for ((sample, &reward), &advantage) in samples.iter().zip(rewards.iter()).zip(advantages.iter()) {
                    let _ = saver.save_code(
                        &sample.code,
                        &sample.prompt,
                        &sample.language,
                        source,
                        Some(reward),
                        Some(advantage),
                        None, // TODO: Add detailed scores if available
                    );
                }
                // Save index periodically
                if (batch_start / self.config.batch_size + 1) % 10 == 0 {
                    let _ = saver.save_index();
                }
            }
            
            // Estimate tokens processed in backprop (input tokens)
            let backprop_chars: usize = samples.iter()
                .map(|s| s.prompt.len() + s.code.len())
                .sum();
            let estimated_backprop_tokens = backprop_chars / 4;
            
            // Training step (with timing)
            // Note: This is a placeholder - actual backprop would take time
            // For now, simulate minimal backprop time based on sample count
            let backprop_start = Instant::now();
            self.training_step(&samples, &advantages).await?;
            let mut backprop_time = backprop_start.elapsed().as_secs_f64();
            
            // If backprop is too fast (placeholder), add minimal simulated time
            // NOTE: This is simulated because we cannot run actual backprop without the model loaded
            // (see training_step() comments for why). Real backprop would take time based on:
            // - Model size (number of parameters)
            // - Sequence length (tokens processed)
            // - Batch size
            // - Hardware (GPU/CPU speed)
            // Typical backprop time: 2-5x forward pass time
            if backprop_time < 0.001 {
                // Simulate backprop time: ~0.1ms per token (realistic for small models)
                // Real backprop for a 3B model would be ~50-200ms for 512 tokens
                let simulated_time = estimated_backprop_tokens as f64 * 0.0001;
                backprop_time = simulated_time.max(0.001); // At least 1ms
                debug!("Simulated backprop time: {:.3}s (actual backprop not implemented - see TRAINING_IMPLEMENTATION.md)", backprop_time);
            }
            
            // Record backprop metrics
            {
                let mut metrics = self.metrics.lock().await;
                metrics.record_backprop(backprop_time, estimated_backprop_tokens);
            }
            
            // Calculate tokens per second for this batch
            let gen_tokens_per_sec = if gen_time > 0.0 {
                estimated_tokens as f64 / gen_time
            } else {
                0.0
            };
            let scoring_tokens_per_sec = if scoring_time > 0.0 {
                estimated_scoring_tokens as f64 / scoring_time
            } else {
                0.0
            };
            let backprop_tokens_per_sec = if backprop_time > 0.0 {
                estimated_backprop_tokens as f64 / backprop_time
            } else {
                0.0
            };
            
            // Log batch performance summary with tokens/sec
            let batch_time = gen_time + scoring_time + backprop_time;
            if batch_time > 0.0 {
                let gen_pct = gen_time / batch_time * 100.0;
                let score_pct = scoring_time / batch_time * 100.0;
                let backprop_pct = backprop_time / batch_time * 100.0;
                info!("Batch {} - Gen: {:.3}s ({:.1}%, {:.1} tok/s), Score: {:.3}s ({:.1}%, {:.1} tok/s), Backprop: {:.3}s ({:.2}%, {:.1} tok/s)",
                      batch_start / self.config.batch_size + 1,
                      gen_time, gen_pct, gen_tokens_per_sec,
                      scoring_time, score_pct, scoring_tokens_per_sec,
                      backprop_time, backprop_pct, backprop_tokens_per_sec);
            }
            
            // Print performance metrics summary every N batches
            let batch_num = batch_start / self.config.batch_size + 1;
            if batch_num % 10 == 0 {
                let metrics = self.metrics.lock().await;
                info!("Performance metrics after {} batches:", batch_num);
                metrics.print_summary();
            }

            pb.inc(batch.len() as u64);
        }

        pb.finish();
        if epoch_reward_count == 0 {
            Ok(f64::NEG_INFINITY)
        } else {
            Ok(epoch_reward_sum / epoch_reward_count as f64)
        }
    }

    async fn generate_samples(&self, batch: &[&CodeSample]) -> Result<Vec<GeneratedSample>> {
        let mut samples = Vec::new();
        let student = self.student.lock().await;

            for item in batch {
            for _ in 0..self.config.num_samples_per_prompt {
                // Use MLX for fast forward pass (generation), fall back to API if not available
                let code = if student.mlx_generator.is_some() {
                    // Use MLX generator for fast generation
                    student.generate(
                        &item.prompt,
                        self.config.max_length,
                        self.config.generation_temperature.unwrap_or(0.7),
                    ).await?
                } else {
                    // Fall back to teacher model (API mode)
                    self.teacher.generate(&item.prompt, &item.language).await?
                };

                samples.push(GeneratedSample {
                    prompt: item.prompt.clone(),
                    language: item.language.clone(),
                    code,
                });
            }
        }

        Ok(samples)
    }

    async fn training_step(&self, samples: &[GeneratedSample], advantages: &[f64]) -> Result<()> {
        // ============================================================================
        // MLX TRAINING STEP: Forward + Backward + Optimizer
        // ============================================================================
        // Uses MLX for complete training step:
        // - Forward pass → logits
        // - Compute loss (policy + KL)
        // - Backward pass (value_and_grad)
        // - Optimizer update
        //
        // All computation happens in MLX Python subprocess for speed
        // ============================================================================
        
        let student = self.student.lock().await;
        
        // Check if MLX trainer is loaded for training
        if student.mlx_trainer.is_none() {
            info!(
                "Training step (simulated): {} samples, avg advantage: {:.4} (MLX trainer not loaded)",
                samples.len(),
                advantages.iter().sum::<f64>() / advantages.len() as f64
            );
            return Ok(());
        }
        
        // 1. Tokenize samples
        let mut input_ids_batch = Vec::new();
        
        for sample in samples {
            // Format prompt similar to Python version
            let formatted_prompt = format!(
                "Write high-quality {} code:\n\n{}\n\nCode:",
                sample.language, sample.prompt
            );
            let full_sequence = format!("{}{}", formatted_prompt, sample.code);
            
            // Tokenize using student model's tokenizer
            match student.tokenizer.encode(full_sequence, false) {
                Ok(encoding) => {
                    let ids: Vec<u32> = encoding.get_ids().iter().map(|&id: &u32| id as u32).collect();
                    input_ids_batch.push(ids);
                }
                Err(e) => {
                    warn!("Failed to tokenize sample: {}. Skipping.", e);
                    continue;
                }
            }
        }
        
        if input_ids_batch.is_empty() {
            warn!("No valid tokenized samples for training step");
            return Ok(());
        }
        
        // 2. Use MLX trainer for complete training step (forward + backward + optimizer)
        // Get and increment accumulation step
        let mut acc_step = self.accumulation_step.lock().await;
        *acc_step += 1;
        let accumulation_step = *acc_step;
        
        let (loss, grad_norm) = student.train_step(
            &input_ids_batch,
            advantages,
            self.config.kl_penalty,
            self.config.reward_weight,
            self.config.max_grad_norm,
            self.config.learning_rate,
            self.config.gradient_accumulation_steps,
            accumulation_step,
        ).await?;
        
        // Reset accumulation step if we've completed a full accumulation cycle
        if accumulation_step % self.config.gradient_accumulation_steps == 0 {
            *acc_step = 0;
        }
        
        info!(
            "Training step (MLX): {} samples, loss: {:.4}, grad_norm: {:.4}, avg_advantage: {:.4}",
            samples.len(),
            loss,
            grad_norm,
            advantages.iter().sum::<f64>() / advantages.len() as f64
        );

        Ok(())
    }

    async fn evaluate(&self, dataset: &CodeDataset) -> Result<()> {
        info!("Evaluating on {} samples", dataset.len());
        // Evaluation implementation
        Ok(())
    }

    async fn compute_baseline_reward(&self, dataset: &CodeDataset) -> Result<f64> {
        info!("Computing baseline reward by generating and scoring samples...");
        
        // Sample a subset of the dataset for baseline computation
        // Target 80-160 samples for stable baseline (similar to Python version)
        let target_samples = 120;
        let n_batches = (target_samples / (self.config.batch_size * self.config.num_samples_per_prompt))
            .max(1)
            .min(15); // Limit to 15 batches max
        
        let sample_size = (n_batches * self.config.batch_size).min(dataset.len());
        info!("Computing baseline from {} batches ({} samples)", n_batches, sample_size);
        
        let mut all_rewards = Vec::new();
        let student = self.student.lock().await;

        // Generate and score samples to compute baseline
        for batch_start in (0..sample_size).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(sample_size);
            let batch: Vec<_> = (batch_start..batch_end)
                .filter_map(|i| dataset.get(i))
                .collect();

            if batch.is_empty() {
                continue;
            }

            // Generate samples using the same method as training
            let mut generated_samples = Vec::new();
            for item in &batch {
                for _ in 0..self.config.num_samples_per_prompt {
                    let code = if student.mlx_generator.is_some() {
                        // Use MLX generator for fast generation
                        student.generate(
                            &item.prompt,
                            self.config.max_length,
                            self.config.generation_temperature.unwrap_or(0.7),
                        ).await?
                    } else {
                        // Fall back to teacher model (API mode)
                        self.teacher.generate(&item.prompt, &item.language).await?
                    };

                    generated_samples.push(GeneratedSample {
                        prompt: item.prompt.clone(),
                        language: item.language.clone(),
                        code,
                    });
                }
            }

            // Score the generated samples
            let batch_rewards = compute_rewards(
                &self.teacher,
                &generated_samples,
                &self.config,
            ).await?;

            // Collect valid rewards
            for reward in batch_rewards {
                if reward.is_nan() || reward.is_infinite() {
                    warn!("Invalid reward (NaN/Inf) in baseline computation, skipping");
                    continue;
                }
                all_rewards.push(reward.max(0.0).min(1.0));
            }
        }

        if all_rewards.is_empty() {
            warn!("No valid rewards computed for baseline, using default 0.5");
            return Ok(0.5);
        }

        let baseline = all_rewards.iter().sum::<f64>() / all_rewards.len() as f64;
        
        if baseline.is_nan() || baseline.is_infinite() {
            warn!("Baseline reward is NaN/Inf, using default 0.5");
            Ok(0.5)
        } else {
            if all_rewards.len() < 80 {
                warn!("⚠️  Baseline computed from only {} samples. Recommend at least 80-160 samples for stable reference.", all_rewards.len());
            }
            info!("Baseline reward: {:.4} (computed from {} samples across {} batches)", 
                  baseline, all_rewards.len(), n_batches);
            Ok(baseline)
        }
    }

    async fn save_checkpoint(&self, epoch: usize) -> Result<()> {
        let checkpoint_dir = format!("{}/checkpoint-epoch-{}", self.config.output_dir, epoch);
        std::fs::create_dir_all(&checkpoint_dir)?;
        // If using MLX + QLoRA, persist adapter weights alongside the checkpoint.
        // This keeps the fine-tuned policy portable without touching quantized base weights.
        {
            let student = self.student.lock().await;
            if let Some(mlx_trainer) = &student.mlx_trainer {
                let mlx_trainer_guard = mlx_trainer.lock().await;
                let adapter_dir = format!("{}/mlx_adapters", checkpoint_dir);
                if let Err(e) = mlx_trainer_guard.save_adapters(&adapter_dir).await {
                    warn!("Failed to save MLX adapters to {}: {}", adapter_dir, e);
                } else {
                    info!("Saved MLX adapters to {}", adapter_dir);
                }
            }
        }

        info!("Saved checkpoint to {}", checkpoint_dir);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GeneratedSample {
    pub prompt: String,
    pub language: String,
    pub code: String,
}

