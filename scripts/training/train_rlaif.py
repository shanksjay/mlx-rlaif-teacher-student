#!/usr/bin/env python3
"""
RLAIF (Reinforcement Learning from AI Feedback) Training Script for Qwen Code Model

This script implements a teacher-student training scheme where:
- Teacher: OpenAI Codex or Claude (provides high-quality code examples)
- Student: Qwen model (being fine-tuned)
- Training: Uses teacher feedback to score and improve student outputs
"""

import os
import json
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import psutil
import threading
import time

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "info"  # Show detailed warnings about generation flags
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    get_scheduler,
)
from datasets import load_dataset
import openai
from anthropic import Anthropic

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress httpx HTTP request logs in INFO mode (only show in DEBUG)
# httpx logs every HTTP request at INFO level, which is too verbose
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)  # Only show WARNING and above (suppress INFO)

# Suppress PyTorch inductor warnings (harmless on Apple Silicon MPS)
# These warnings are CUDA-specific and don't apply to MPS
import warnings
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm mode.*")
warnings.filterwarnings("ignore", message=".*max_autotune_gemm.*")
# Suppress TensorBoard pkg_resources deprecation warning
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)
# Also suppress via logging for torch._inductor
torch_inductor_logger = logging.getLogger("torch._inductor.utils")
torch_inductor_logger.setLevel(logging.ERROR)  # Only show errors, suppress warnings


@dataclass
class RLAIFConfig:
    """Configuration for RLAIF training"""
    base_model: str
    teacher_provider: str
    teacher_model: str
    teacher_api_key_env: str
    output_dir: str
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    save_steps: int
    eval_steps: int
    logging_steps: int
    save_every_epochs: int = 1
    save_every_batches: int = 0
    max_grad_norm: float
    weight_decay: float
    lr_scheduler_type: str
    reward_weight: float
    kl_penalty: float
    beta: float
    num_samples_per_prompt: int
    max_length: int
    use_4bit: bool
    use_mps: bool
    mixed_precision: str
    tensorboard_dir: str
    log_level: str
    save_json_summaries: bool = True
    json_summaries_dir: str = "./logs/json_summaries"
    baseline_eval_batches: int = 1
    tensorboard_batch_interval: int = 1
    top_k: int = 50
    top_p: float = 0.95
    generation_temperature: float = 0.8  # Temperature for generation (higher = more exploration)
    curriculum_learning: bool = False  # Enable curriculum learning
    reward_bonuses: bool = False  # Enable reward bonuses for specific improvements
    use_lora: bool = False  # Use LoRA for efficient fine-tuning
    use_qlora: bool = False  # Use QLoRA (4-bit + LoRA) for maximum efficiency
    lora_r: int = 16  # LoRA rank (higher = more parameters, better quality)
    lora_alpha: int = 32  # LoRA alpha (scaling factor, typically 2x rank)
    lora_dropout: float = 0.05  # LoRA dropout
    lora_target_modules: Optional[List[str]] = None  # Target modules for LoRA (None = auto-detect)
    save_mlx_format: bool = True
    mlx_quantization: Optional[str] = None
    upload_to_hub: bool = False
    hf_repo_id: Optional[str] = None
    hf_token_env: str = "HUGGINGFACE_TOKEN"
    upload_quantized: bool = True
    hf_private: bool = False
    upload_datasets: bool = True
    dataset_repo_id: Optional[str] = None
    save_datasets_locally: bool = True
    dataset_output_dir: str = "./datasets"
    use_safetensors: bool = True
    low_cpu_mem_usage: bool = True
    use_mlx_for_generation: bool = True  # Use MLX for faster generation (requires MLX model) - enabled by default
    mlx_model_path: Optional[str] = None  # Path to MLX model for generation
    # CUDA/Unsloth (optional): enables Unsloth optimized model loading/training/generation on NVIDIA GPUs
    use_unsloth: bool = False
    unsloth_dtype: str = "bf16"  # "bf16" or "fp16"
    unsloth_max_seq_length: Optional[int] = None  # If None, falls back to max_length


class CodeDataset(Dataset):
    """Dataset for code training examples"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Only load from file if data_file is provided and not empty
        if data_file and data_file.strip():
            logger.info(f"Loading dataset from {data_file}")
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Dataset file not found: {data_file}")
            with open(data_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
            logger.info(f"Loaded {len(self.data)} examples")
        else:
            # Empty dataset - data can be set directly (e.g., for curriculum learning)
            logger.debug("Created empty dataset (data will be set directly)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get('prompt', '')
        language = item.get('language', 'python')
        
        # Format prompt with language context
        formatted_prompt = f"Write high-quality {language} code:\n\n{prompt}\n\nCode:"
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'prompt': prompt,
            'language': language,
            'prompt_text': formatted_prompt
        }


class TeacherModel:
    """Wrapper for teacher models (OpenAI or Anthropic)"""
    
    def __init__(self, provider: str, model_name: str, api_key_env: str, temperature: float = 0.7):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        
        api_key = os.getenv(api_key_env)
        if not api_key:
            error_msg = (
                f"\n{'='*80}\n"
                f"ERROR: API key not found in environment variable '{api_key_env}'\n"
                f"{'='*80}\n"
                f"Please set your API key:\n"
                f"  export {api_key_env}='your-api-key'\n\n"
                f"For OpenAI:\n"
                f"  export OPENAI_API_KEY='sk-...'\n"
                f"  Get your key from: https://platform.openai.com/api-keys\n\n"
                f"For Anthropic:\n"
                f"  export ANTHROPIC_API_KEY='sk-ant-...'\n"
                f"  Get your key from: https://console.anthropic.com/\n"
                f"{'='*80}\n"
            )
            raise ValueError(error_msg)
        
        if provider == "openai":
            openai.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key)
        elif provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
            # Test model availability and try fallbacks if needed
            self._test_and_fallback_model()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _test_and_fallback_model(self):
        """Test if the model is available, try fallbacks if not"""
        # List of models to try in order (newer, non-deprecated models first)
        models_to_try = [self.model_name]  # Start with requested model
        fallback_models = [
            "claude-3-5-haiku-20241022",  # Newest, fastest, cheapest (recommended)
            "claude-3-5-sonnet-20241022",  # Newest, better quality
            "claude-3-opus-20240229",  # Deprecated but may still work
            "claude-3-sonnet-20240229",  # Deprecated but may still work
        ]
        
        # Add fallbacks that aren't already the requested model
        for fallback in fallback_models:
            if fallback != self.model_name:
                models_to_try.append(fallback)
        
        # Suppress deprecation warnings during model testing
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*deprecated.*")
            
            # Try each model until one works
            for model_name in models_to_try:
                try:
                    # Minimal test call
                    test_response = self.client.messages.create(
                        model=model_name,
                        max_tokens=1,
                        messages=[{"role": "user", "content": "test"}]
                    )
                    if model_name != self.model_name:
                        logger.warning(f"⚠️  Model '{self.model_name}' not available. Using fallback: '{model_name}'")
                    else:
                        logger.info(f"✓ Model '{model_name}' is available")
                    self.model_name = model_name
                    return
                except Exception as e:
                    error_str = str(e)
                    if "404" in error_str or "not_found" in error_str.lower():
                        continue  # Try next model
                    else:
                        # Other errors (rate limit, auth, etc.) - log and continue with first model
                        logger.warning(f"⚠️  Error testing model '{model_name}': {e}")
                        if model_name == self.model_name:
                            # If the requested model fails for non-404 reasons, continue
                            continue
        
        # If all models fail, log error but continue (will fail on actual use with better error)
        logger.error(f"❌ Could not find any available Anthropic model.")
        logger.error(f"   Tried: {', '.join(models_to_try)}")
        logger.error(f"   Check your API key permissions at: https://console.anthropic.com/settings/keys")
        logger.error(f"   Your API key may not have access to these models.")
    
    def generate(self, prompt: str, language: str, max_tokens: int = 2048) -> str:
        """Generate code using teacher model"""
        system_prompt = f"You are an expert {language} programmer. Generate clean, efficient, and well-documented code."
        full_prompt = f"{prompt}\n\nGenerate high-quality {language} code:"
        
        # Suppress deprecation warnings for Anthropic API calls
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*deprecated.*")
            
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": full_prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=max_tokens
                    )
                    return response.choices[0].message.content.strip()
                
                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": full_prompt}]
                    )
                    # Track API token usage if available
                    if hasattr(response, 'usage') and hasattr(response.usage, 'input_tokens'):
                        # Store in trainer if available (will be set by trainer)
                        if hasattr(self, '_trainer_ref') and self._trainer_ref:
                            self._trainer_ref.training_metrics['api_tokens_sent'] += response.usage.input_tokens
                    return response.content[0].text.strip()
            
            except Exception as e:
                # Count teacher API failures as scoring/generation errors in trainer stats.
                # (These errors were previously swallowed, which made epoch summaries misleading.)
                try:
                    if hasattr(self, '_trainer_ref') and self._trainer_ref:
                        self._trainer_ref.error_stats['teacher_generate_errors'] += 1
                        self._trainer_ref.error_stats['scoring_errors'] += 1
                except Exception:
                    pass
                error_str = str(e)
                if "404" in error_str or "not_found" in error_str.lower():
                    logger.error(f"Error: Model '{self.model_name}' not found. This may be due to:")
                    logger.error(f"  1. Incorrect model name format")
                    logger.error(f"  2. Model not available in your API plan")
                    logger.error(f"  3. API key doesn't have access to this model")
                    logger.error(f"  Common Anthropic models: 'claude-3-5-sonnet', 'claude-3-5-haiku', 'claude-3-opus-20240229'")
                    logger.error(f"  Check available models at: https://docs.anthropic.com/claude/docs/models-overview")
                else:
                    logger.error(f"Error generating from teacher model: {e}")
                return ""
    
    def score_code(self, code: str, prompt: str, language: str, use_cache: bool = True) -> float:
        """Score code quality using teacher model with optional caching"""
        # Cache key for scoring (include code hash to avoid collisions)
        if use_cache:
            import hashlib
            code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
            cache_key = f"{code_hash}:{prompt}:{language}"
            # Check cache (if we have a cache available)
            # Note: We'll implement this in the trainer class
        
        scoring_prompt = f"""Evaluate the following {language} code on a scale of 0.0 to 1.0 based on:
1. Correctness (0.3): Does it solve the problem correctly?
2. Code Quality (0.3): Is it clean, readable, and well-structured?
3. Efficiency (0.2): Is it efficient and follows best practices?
4. Documentation (0.2): Is it well-documented?

Prompt: {prompt}

Code:
```{language}
{code}
```

IMPORTANT: Respond with ONLY a single float between 0.0 and 1.0 (e.g., 0.75). Do not include explanations, additional text, or newlines. Just the number."""
        
        # Suppress deprecation warnings for Anthropic API calls
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*deprecated.*")
            
            try:
                api_input_tokens = 0
                api_output_tokens = 0
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": scoring_prompt}],
                        temperature=0.1,
                        max_tokens=50
                    )
                    score_text = response.choices[0].message.content.strip()
                    # Track API token usage (input and output separately)
                    if hasattr(response, 'usage'):
                        if hasattr(response.usage, 'prompt_tokens'):
                            api_input_tokens = response.usage.prompt_tokens
                        if hasattr(response.usage, 'completion_tokens'):
                            api_output_tokens = response.usage.completion_tokens
                        if hasattr(self, '_trainer_ref') and self._trainer_ref:
                            self._trainer_ref.training_metrics['api_tokens_sent'] += api_input_tokens
                            self._trainer_ref.training_metrics['api_tokens_received'] += api_output_tokens
                else:  # anthropic
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=50,
                        temperature=0.1,
                        messages=[{"role": "user", "content": scoring_prompt}]
                    )
                    score_text = response.content[0].text.strip()
                    # Track API token usage (input and output separately)
                    if hasattr(response, 'usage'):
                        if hasattr(response.usage, 'input_tokens'):
                            api_input_tokens = response.usage.input_tokens
                        if hasattr(response.usage, 'output_tokens'):
                            api_output_tokens = response.usage.output_tokens
                        if hasattr(self, '_trainer_ref') and self._trainer_ref:
                            self._trainer_ref.training_metrics['api_tokens_sent'] += api_input_tokens
                            self._trainer_ref.training_metrics['api_tokens_received'] += api_output_tokens
                
                # Extract float from response (handle cases where score is embedded in text)
                import re
                try:
                    # Try direct float conversion first
                    score = float(score_text.strip())
                    return max(0.0, min(1.0, score))  # Clamp to [0, 1]
                except ValueError:
                    # If that fails, try to extract first float from the text
                    # Look for patterns like "0.4", "0.75", "1.0", etc.
                    float_pattern = r'\b(0\.\d+|1\.0|0|1)\b'
                    matches = re.findall(float_pattern, score_text)
                    if matches:
                        try:
                            score = float(matches[0])
                            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
                        except ValueError:
                            pass
                    
                    # If still no match, try to find any number between 0 and 1
                    # Look for decimal numbers in the range [0, 1]
                    decimal_pattern = r'\b(0?\.\d+)\b'
                    decimal_matches = re.findall(decimal_pattern, score_text)
                    if decimal_matches:
                        try:
                            score = float(decimal_matches[0])
                            if 0.0 <= score <= 1.0:
                                return score
                        except ValueError:
                            pass
                    
                    logger.warning(f"Could not parse score from: {score_text[:100]}...")
                    return 0.5
            
            except Exception as e:
                # Count teacher scoring API failures so epoch summary reflects reality.
                # We still return a default score so training can continue, but we must record the failure.
                try:
                    if hasattr(self, '_trainer_ref') and self._trainer_ref:
                        self._trainer_ref.error_stats['teacher_scoring_errors'] += 1
                        self._trainer_ref.error_stats['scoring_errors'] += 1
                except Exception:
                    pass
                error_str = str(e)
                if "404" in error_str or "not_found" in error_str.lower():
                    logger.error(f"Error: Model '{self.model_name}' not found when scoring code.")
                    logger.error(f"  Check your model name in config.yaml. Common models: 'claude-3-5-sonnet', 'claude-3-5-haiku'")
                else:
                    logger.error(f"Error scoring code: {e}")
                return 0.5


class RLAIFTrainer:
    """RLAIF Trainer implementing teacher-student training"""
    
    def _init_json_summaries(self):
        """Initialize JSONL summary files for offline analysis."""
        try:
            if not getattr(self.config, "save_json_summaries", True):
                return
            out_dir = getattr(self.config, "json_summaries_dir", "./logs/json_summaries") or "./logs/json_summaries"
            os.makedirs(out_dir, exist_ok=True)
            self._json_summaries_dir = out_dir
            self._batch_jsonl_path = os.path.join(out_dir, "batches.jsonl")
            self._epoch_jsonl_path = os.path.join(out_dir, "epochs.jsonl")
            logger.info(f"JSON summaries enabled -> {out_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize JSON summaries: {e}")
            self._json_summaries_dir = None
            self._batch_jsonl_path = None
            self._epoch_jsonl_path = None

    def _append_jsonl(self, path: Optional[str], payload: Dict):
        """Append a single JSON record to a JSONL file."""
        if not path:
            return
        try:
            payload = dict(payload)
            payload.setdefault("ts_unix", time.time())
            payload.setdefault("ts_iso", datetime.utcnow().isoformat() + "Z")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write JSONL to {path}: {e}")

    def _log_batch_json(self, payload: Dict):
        if getattr(self.config, "save_json_summaries", True):
            self._append_jsonl(self._batch_jsonl_path, payload)

    def _log_epoch_json(self, payload: Dict):
        if getattr(self.config, "save_json_summaries", True):
            self._append_jsonl(self._epoch_jsonl_path, payload)

    def _compute_baseline_reward(self, train_loader: DataLoader) -> float:
        """Compute a pre-training baseline reward (no weight updates).
        
        Uses the first N batches (config.logging.baseline_eval_batches) to estimate baseline quality.
        """
        n_batches = int(getattr(self.config, "baseline_eval_batches", 0) or 0)
        if n_batches <= 0:
            return 0.0

        logger.info(f"Computing baseline reward on first {n_batches} batch(es) (no training updates)...")
        all_rewards: List[float] = []

        # Snapshot caches to avoid polluting training caches too much
        old_cache = dict(self.teacher_score_cache) if hasattr(self, "teacher_score_cache") else {}
        try:
            it = iter(train_loader)
            for bi in range(n_batches):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                prompts = batch.get("prompt")
                languages = batch.get("language")
                if not prompts or not languages:
                    continue
                try:
                    samples = self.generate_student_samples(prompts, languages, num_samples=self.config.num_samples_per_prompt, epoch=0)
                    if samples:
                        rewards, _ = self.compute_rewards(samples, save_to_dataset=False)
                        if rewards:
                            all_rewards.extend([float(r) for r in rewards])
                except Exception as e:
                    logger.warning(f"Baseline batch {bi} failed: {e}")
                    continue
        finally:
            # Restore cache and let epoch logic manage fresh scoring
            try:
                self.teacher_score_cache = old_cache
            except Exception:
                pass

        baseline = float(np.mean(all_rewards)) if all_rewards else 0.0
        logger.info(f"Baseline reward: {baseline:.4f} (computed from {len(all_rewards)} samples)")
        return baseline
    
    def __init__(self, config: RLAIFConfig):
        self.config = config
        self.device = self._setup_device()
        self._unsloth_enabled = False
        self._unsloth_flm = None  # set to unsloth.FastLanguageModel when available
        self._json_summaries_dir = None
        self._batch_jsonl_path = None
        self._epoch_jsonl_path = None
        self._init_json_summaries()
        self.baseline_reward = None
        self._prev_batch_avg_reward = None
        self._reward_ema = None  # exponential moving average of batch avg reward
        self._batch_step = 0  # monotonic batch counter for continuous TensorBoard time series
        
        # Load model and tokenizer
        logger.info(f"Loading base model: {config.base_model}")

        # Pre-init monitoring vars so later analysis code doesn't crash in Unsloth mode
        load_start = time.time()
        process = None
        mem_before = 0.0
        memory_samples = []
        load_time = 0.0
        shard_time = 0.0

        # Optional: Unsloth (CUDA-only) for NVIDIA benchmarking
        if config.use_unsloth:
            if self.device.type != "cuda":
                # Avoid accidental "slow CPU run" when user copied an NVIDIA config onto a Mac.
                # Auto-disable Unsloth and continue with standard loading; provide an actionable hint.
                logger.info("Unsloth requested but CUDA is not available -> disabling Unsloth and continuing with standard Transformers loading.")
                if torch.backends.mps.is_available() and getattr(config, "use_mps", False) is False:
                    logger.info("Tip: you're on Apple Silicon; set `hardware.use_mps: true` (and optionally `hardware.use_mlx_for_generation: true`).")
                config.use_unsloth = False
            else:
                try:
                    from unsloth import FastLanguageModel  # type: ignore
                    self._unsloth_flm = FastLanguageModel
                    max_seq_len = config.unsloth_max_seq_length or config.max_length
                    dtype_str = (config.unsloth_dtype or "bf16").lower()
                    dtype = torch.bfloat16 if dtype_str in ("bf16", "bfloat16") else torch.float16

                    logger.info("Loading model via Unsloth (CUDA)...")
                    # Unsloth returns a HF-compatible model/tokenizer, but much faster on NVIDIA.
                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_name=config.base_model,
                        max_seq_length=max_seq_len,
                        dtype=dtype,
                        load_in_4bit=bool(config.use_4bit),
                    )
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    self._unsloth_enabled = True
                    load_time = time.time() - load_start
                    logger.info(f"✓ Unsloth load complete in {load_time:.1f}s (dtype={dtype_str}, 4bit={bool(config.use_4bit)})")
                except Exception as e:
                    logger.warning(f"Failed to load Unsloth. Falling back to standard Transformers loading. Error: {e}")
                    self._unsloth_enabled = False
                    self._unsloth_flm = None
        
        if not self._unsloth_enabled:
            # Load tokenizer first (fast)
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.base_model,
                use_fast=True,  # Use fast tokenizer
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Optimize model loading (matching preload_model.py for speed)
            logger.info("Loading model weights (this may take a few minutes)...")
            load_start = time.time()
            
            # Monitor memory before loading
            try:
                import psutil
                process = psutil.Process()
                mem_before = process.memory_info().rss / (1024 ** 3)
                logger.info(f"Memory before loading: {mem_before:.2f} GB")
            except:
                process = None
                mem_before = 0.0
            
            # Use safetensors for faster loading (same as preload_model.py)
            model_kwargs = {
                "device_map": "auto",
                "dtype": torch.bfloat16 if config.use_mps else torch.float32,  # Use dtype instead of torch_dtype
                "trust_remote_code": True,
                "low_cpu_mem_usage": config.low_cpu_mem_usage,  # Optimize memory usage during loading
            }
            
            # Prefer safetensors format for faster loading
            if config.use_safetensors:
                try:
                    from safetensors.torch import load_file
                    model_kwargs["use_safetensors"] = True
                    logger.info("Using safetensors format for faster loading")
                except ImportError:
                    logger.info("Using standard format (safetensors not available)")
            
            # Setup quantization for M5 MacBook
            # WARNING: 4-bit quantization with bfloat16 on MPS can cause NaN logits
            # BitsAndBytes quantization may not be fully compatible with MPS
            if config.use_4bit:
                if config.use_mps and torch.backends.mps.is_available():
                    logger.warning("⚠️  4-bit quantization on MPS may cause NaN logits!")
                    logger.warning("  BitsAndBytes quantization has known issues with MPS backend.")
                    logger.warning("  Consider using float32 instead of bfloat16, or disable quantization.")
                    logger.warning("  If you see NaN logits, set model.use_4bit: false in config.yaml")
                
                logger.info("Using 4-bit quantization...")
                # Use float32 compute dtype on MPS to avoid numerical instability
                compute_dtype = torch.float32 if (config.use_mps and torch.backends.mps.is_available()) else torch.bfloat16
                if compute_dtype == torch.float32:
                    logger.info("  Using float32 compute dtype on MPS for stability (slower but more stable)")
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,  # Use float32 on MPS for stability
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["dtype"] = compute_dtype  # Match compute dtype
            
            # Simplified monitoring (matching preload_model.py for less overhead)
            monitoring = True
            memory_samples = []
            shard_start = time.time()
            
            def monitor_loading():
                """Monitor memory during loading (simplified for speed)"""
                while monitoring:
                    elapsed = time.time() - shard_start
                    try:
                        if process:
                            process_mem = process.memory_info().rss / (1024 ** 3)
                            system_mem = psutil.virtual_memory().used / (1024 ** 3)
                            # Store both process and system memory
                            memory_samples.append((elapsed, process_mem, system_mem))
                    except:
                        pass
                    time.sleep(0.5)  # Sample every 0.5 seconds (less frequent = less overhead)
            
            monitor_thread = threading.Thread(target=monitor_loading, daemon=True)
            monitor_thread.start()
            
            # Clear cache before loading
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load model (same pattern as preload_model.py)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.base_model,
                    **model_kwargs
                )
            except Exception as e:
                logger.warning(f"Error with optimized loading: {e}. Trying standard loading...")
                # Fallback to standard loading
                if config.use_4bit:
                    # Use float32 compute dtype on MPS for stability
                    compute_dtype = torch.float32 if (config.use_mps and torch.backends.mps.is_available()) else torch.bfloat16
                    # Update bnb_config with correct compute dtype
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=True,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        config.base_model,
                        quantization_config=bnb_config,
                        device_map="auto",
                        dtype=compute_dtype,  # Match compute dtype
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        config.base_model,
                        device_map="auto",
                        dtype=torch.bfloat16 if config.use_mps else torch.float32,  # Use dtype instead of torch_dtype
                    )
            
            monitoring = False
            monitor_thread.join(timeout=2)
            
            load_time = time.time() - load_start
            shard_time = time.time() - shard_start
        
        # Apply LoRA/QLoRA if enabled
        if config.use_lora or config.use_qlora:
            # Prefer Unsloth-native LoRA when available (CUDA-only). Fallback to PEFT otherwise.
            if self._unsloth_enabled and self._unsloth_flm is not None:
                try:
                    logger.info("Applying LoRA via Unsloth...")
                    target_modules = config.lora_target_modules
                    if target_modules is None:
                        # Reasonable default for many decoder-only models (can be overridden in config)
                        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                    self.model = self._unsloth_flm.get_peft_model(
                        self.model,
                        r=int(config.lora_r),
                        target_modules=target_modules,
                        lora_alpha=int(config.lora_alpha),
                        lora_dropout=float(config.lora_dropout),
                        bias="none",
                        use_gradient_checkpointing="unsloth",
                    )
                except Exception as e:
                    logger.warning(f"Unsloth LoRA setup failed; falling back to PEFT. Error: {e}")
                    self.model = self._apply_lora(config)
            else:
                self.model = self._apply_lora(config)
        
        # Validate model was loaded correctly - check for NaN/Inf in weights
        logger.info("Validating model weights for corruption...")
        corrupted_params = []
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if torch.isnan(param).any():
                nan_count = torch.isnan(param).sum().item()
                corrupted_params.append((name, "NaN", nan_count, param.shape))
            if torch.isinf(param).any():
                inf_count = torch.isinf(param).sum().item()
                corrupted_params.append((name, "Inf", inf_count, param.shape))
        
        if corrupted_params:
            logger.error(f"⚠️  CRITICAL: Model weights contain NaN/Inf values!")
            logger.error(f"  Found {len(corrupted_params)} parameters with corrupted values")
            for name, issue_type, count, shape in corrupted_params[:5]:  # Show first 5
                logger.error(f"    {name}: {issue_type} ({count} values, shape: {shape})")
            logger.error("  This will cause NaN logits during training!")
            logger.error("  SOLUTION: Reload the model or use a different checkpoint.")
        else:
            logger.info(f"✓ Model weights validated - no NaN/Inf detected ({total_params:,} parameters checked)")
        
        # Analyze loading performance (simplified, matching preload_model.py)
        if memory_samples and len(memory_samples) > 4 and process:
            logger.info("\n" + "="*60)
            logger.info("Loading Performance Analysis:")
            logger.info("="*60)
            
            quarter_size = len(memory_samples) // 4
            quarter_times = []
            
            for i in range(4):
                start_idx = i * quarter_size
                end_idx = (i + 1) * quarter_size if i < 3 else len(memory_samples)
                
                if start_idx < len(memory_samples):
                    quarter_data = memory_samples[start_idx:end_idx]
                    if quarter_data:
                        time_start = quarter_data[0][0]
                        time_end = quarter_data[-1][0]
                        # Handle both old format (2 values) and new format (3 values)
                        if len(quarter_data[0]) == 3:
                            process_mem_start = quarter_data[0][1]
                            process_mem_end = quarter_data[-1][1]
                            system_mem_start = quarter_data[0][2]
                            system_mem_end = quarter_data[-1][2]
                            process_mem_peak = max(m[1] for m in quarter_data)
                            system_mem_peak = max(m[2] for m in quarter_data)
                            process_mem_delta = process_mem_end - process_mem_start
                            system_mem_delta = system_mem_end - system_mem_start
                        else:
                            # Old format compatibility
                            process_mem_start = quarter_data[0][1]
                            process_mem_end = quarter_data[-1][1]
                            system_mem_start = system_mem_end = system_mem_peak = 0
                            process_mem_peak = max(m[1] for m in quarter_data)
                            process_mem_delta = process_mem_end - process_mem_start
                            system_mem_delta = 0
                        
                        quarter_time = time_end - time_start
                        quarter_times.append(quarter_time)
                        
                        logger.info(
                            f"\n{i*25}%-{(i+1)*25}% (Shard {i+1}): "
                            f"Time={quarter_time:.1f}s"
                        )
                        logger.info(
                            f"  Process Memory: {process_mem_start:.2f}GB→{process_mem_end:.2f}GB "
                            f"(peak: {process_mem_peak:.2f}GB, Δ{process_mem_delta:+.2f}GB)"
                        )
                        if system_mem_delta != 0:
                            logger.info(
                                f"  System Memory: {system_mem_start:.2f}GB→{system_mem_end:.2f}GB "
                                f"(peak: {system_mem_peak:.2f}GB, Δ{system_mem_delta:+.2f}GB)"
                            )
            
            # Compare quarters
            if len(quarter_times) == 4:
                avg_first_three = sum(quarter_times[:3]) / 3
                last_quarter = quarter_times[3]
                if last_quarter > avg_first_three * 1.2:
                    ratio = last_quarter / avg_first_three
                    logger.info(f"\n⚠️  Last shard (75-100%) is {ratio:.1f}x slower than average")
                    logger.info("   This is normal due to:")
                    logger.info("   - Memory pressure from previous shards")
                    logger.info("   - Device mapping finalization")
                    logger.info("   - Quantization setup completion")
        
        # Monitor memory after loading
        if process:
            try:
                process_mem_after = process.memory_info().rss / (1024 ** 3)
                system_mem_after = psutil.virtual_memory().used / (1024 ** 3)
                process_mem_used = process_mem_after - mem_before
                system_mem_used = system_mem_after - (psutil.virtual_memory().used / (1024 ** 3) if hasattr(psutil, 'virtual_memory') else 0)
                
                logger.info(f"\nMemory after loading:")
                logger.info(f"  Process RSS: {process_mem_after:.2f} GB (Δ{process_mem_used:+.2f} GB)")
                logger.info(f"  System used: {system_mem_after:.2f} GB / {psutil.virtual_memory().total / (1024 ** 3):.2f} GB (Δ{system_mem_used:+.2f} GB)")
                logger.info(f"Shard loading time: {shard_time:.1f}s ({shard_time/60:.1f} minutes)")
            except:
                pass
        
        logger.info(f"\nTotal loading time: {load_time:.1f} seconds ({load_time/60:.1f} minutes)")
        logger.info("="*60)
        
        # Clear cache after loading
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Compile model for M5 if available (PyTorch 2.0+)
        if config.use_mps and hasattr(torch, 'compile'):
            try:
                logger.info("Compiling model with torch.compile for M5 optimization...")
                # Suppress inductor warnings during compilation (harmless on MPS/Apple Silicon)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Not enough SMs.*")
                    warnings.filterwarnings("ignore", message=".*max_autotune.*")
                    # Also suppress via logging
                    torch_inductor_logger = logging.getLogger("torch._inductor.utils")
                    original_level = torch_inductor_logger.level
                    torch_inductor_logger.setLevel(logging.ERROR)
                    try:
                        self.model = torch.compile(self.model, mode="reduce-overhead")
                    finally:
                        torch_inductor_logger.setLevel(original_level)
                logger.info("✓ Model compilation successful")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}. Continuing without compilation.")
        
        # Warm up model for faster first generation
        # First generation is often slow due to MPS initialization and compilation overhead
        logger.info("Warming up model (first generation is slower)...")
        warmup_prompt = "Test"
        warmup_inputs = self.tokenizer(warmup_prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        warmup_inputs = {k: v.to(device) for k, v in warmup_inputs.items()}
        
        with torch.no_grad():
            try:
                # Very short warmup generation to initialize MPS and compilation
                # Explicitly unset sampling parameters when do_sample=False to avoid warnings
                _ = self.model.generate(
                    **warmup_inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=None,  # Unset when do_sample=False
                    top_p=None,  # Unset when do_sample=False
                    top_k=None,  # Unset when do_sample=False
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                logger.info("✓ Model warmed up")
            except Exception as e:
                logger.debug(f"Warmup failed (non-critical): {e}")
        
        # Clear cache after warmup
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize teacher model
        logger.info(f"Initializing teacher model: {config.teacher_provider}/{config.teacher_model}")
        self.teacher = TeacherModel(
            provider=config.teacher_provider,
            model_name=config.teacher_model,
            api_key_env=config.teacher_api_key_env
        )
        # Set reference to trainer for token tracking
        self.teacher._trainer_ref = self
        
        # Setup TensorBoard
        if config.tensorboard_dir:
            os.makedirs(config.tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(config.tensorboard_dir)
        else:
            self.writer = None
        
        # Training stats
        self.stats = {
            'step': 0,
            'epoch': 0,
            'total_reward': 0.0,
            'avg_reward': 0.0,
            'total_loss': 0.0,
            'avg_loss': 0.0,
            'kl_divergence': 0.0,
            'num_samples': 0,
        }
        
        # Dataset collection for Hugging Face upload
        self.dataset_collection = {
            'training': [],
            'validation': [],
            'evaluation': []
        }
        
        # System monitoring
        self.monitoring_enabled = True
        self.monitoring_thread = None
        self.monitoring_interval = 5  # seconds
        
        # Performance optimizations
        self.teacher_cache = {}  # Cache teacher responses (key: f"{prompt}:{language}")
        self.teacher_score_cache = {}  # Cache teacher scores (key: f"{code}:{prompt}:{language}")
        self.executor = ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 4))  # Concurrent API calls (reduced for M5)
        self.use_batch_generation = True  # Batch student generation
        self.generation_warmup_done = False  # Track if generation has been warmed up

        # Cross-epoch diversity tracking (in-memory):
        # - `_epoch_code_hashes` resets at the start of each epoch
        # - `_global_code_hashes` accumulates across epochs in this run
        self._epoch_code_hashes = set()
        self._global_code_hashes = set()
        
        # MPS memory management: Set high watermark ratio to allow more memory usage
        # This helps prevent OOM errors on systems with unified memory
        if config.use_mps and torch.backends.mps.is_available():
            # Set environment variable to allow more memory allocation
            # 0.0 = no limit (use with caution), 0.8 = 80% of available memory
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
            logger.info("MPS memory management: Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to allow more memory")
            # Also reduce per-process memory fraction to leave more headroom
            # Check if method exists (not available in all PyTorch versions)
            if hasattr(torch.backends.mps, 'set_per_process_memory_fraction'):
                torch.backends.mps.set_per_process_memory_fraction(0.6)  # Reduced from 0.7 to 0.6
                logger.info("MPS memory: Set per-process memory fraction to 0.6 (60%)")
            else:
                logger.info("MPS memory: set_per_process_memory_fraction not available in this PyTorch version")
        
        # MLX model for faster generation (optional, much faster than PyTorch MPS)
        # Load MLX model if enabled (similar to preload_model.py)
        self.mlx_model = None
        self.mlx_tokenizer = None
        
        # Training metrics tracking
        self.training_metrics = {
            'generation_tokens_per_sec': [],  # Track all generation speeds
            'backprop_tokens_per_sec': [],    # Track all backprop speeds
            'api_tokens_sent': 0,             # Total tokens sent to teacher API (input tokens)
            'api_tokens_received': 0,         # Total tokens received from teacher API (output tokens)
            'api_tokens_by_epoch': [],        # Tokens per epoch (input tokens)
            'api_output_tokens_by_epoch': [], # Output tokens per epoch
            'api_calls_by_epoch': [],         # Number of API calls per epoch
            'cache_hits_by_epoch': [],        # Number of cache hits per epoch
            'scoring_breakdown_by_epoch': [], # Scoring breakdown per epoch: [{'correctness': avg, 'code_quality': avg, 'efficiency': avg, 'documentation': avg}, ...]
            'reward_by_epoch': [],            # Average reward per epoch (for trend analysis)
            'loss_by_epoch': [],              # Average loss per epoch (for trend analysis)
            'reward_variance_by_epoch': [],   # Reward variance per epoch (lower is better)
            'code_diversity_by_epoch': [],    # Code diversity metrics per epoch
            'training_start_time': None,      # Training start time
            'training_end_time': None,        # Training end time
        }
        
        # Cache statistics
        self.cache_stats = {
            # Teacher reference generation (teacher.generate) caching
            'teacher_gen_calls': 0,
            'teacher_gen_cache_hits': 0,
            # Teacher scoring (teacher.score_code) caching (trainer-level)
            'teacher_score_calls': 0,
            'teacher_score_cache_hits': 0,
            # Back-compat / aggregated counters (legacy)
            'api_calls': 0,      # historically used for teacher generation calls
            'cache_hits': 0,     # historically used for teacher generation cache hits
            'cache_misses': 0,   # historically used for teacher generation cache misses
        }
        
        # Error statistics
        self.error_stats = {
            'generation_errors': 0,      # Errors during student generation
            # Back-compat aggregate: total teacher API failures (generate + score) and any reward-thread scoring failures
            'scoring_errors': 0,
            # Split counters (requested)
            'teacher_generate_errors': 0,
            'teacher_scoring_errors': 0,
            'generation_errors_by_epoch': [],  # Generation errors per epoch
            'scoring_errors_by_epoch': [],     # Scoring errors per epoch
            'teacher_generate_errors_by_epoch': [],
            'teacher_scoring_errors_by_epoch': [],
        }
        if config.use_unsloth and config.use_mlx_for_generation:
            logger.info("Unsloth enabled; skipping MLX generation (MLX is Apple Silicon only).")

        if config.use_mlx_for_generation and not config.use_unsloth:
            # Auto-detect MLX model path if not specified
            mlx_path = config.mlx_model_path
            if mlx_path is None:
                # Try common MLX model locations in order of preference
                possible_paths = [
                    "./mlx_model/q4",  # Q4 quantized (fastest + smallest; best throughput)
                    "./mlx_model/q8",  # Q8 quantized (best balance)
                    "./mlx_model/base", # Unquantized base model
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        mlx_path = path
                        logger.info(f"Auto-detected MLX model at: {mlx_path}")
                        break
                if mlx_path is None:
                    logger.warning("MLX enabled but no model found. Expected locations:")
                    for path in possible_paths:
                        logger.warning(f"  - {path}")
                    logger.warning("To convert model to MLX:")
                    logger.warning(f"  uv run python scripts/utils/convert_to_mlx.py --hf-path {self.config.base_model} --mlx-path ./mlx_model/q8 --quantize q8_bit")
                    logger.warning("Falling back to PyTorch MPS for generation (slower)")
            
            # Load MLX model if path was found or specified
            if mlx_path is not None:
                self._load_mlx_model_for_generation(mlx_path)
        elif config.mlx_model_path:
            # MLX path specified but not enabled - warn user
            logger.info(f"MLX model path specified ({config.mlx_model_path}) but use_mlx_for_generation is False")
            logger.info("To use MLX for faster generation, set in config.yaml:")
            logger.info("  hardware:")
            logger.info("    use_mlx_for_generation: true")
            logger.info(f"    mlx_model_path: {config.mlx_model_path}")
    
    def _setup_device(self):
        """Setup device for training/generation.
        
        Notes:
        - Apple Silicon: prefers MPS when enabled and available.
        - NVIDIA: if `use_unsloth` is enabled, we strongly prefer CUDA.
        """
        # If user explicitly wants Unsloth, we must be on CUDA.
        if getattr(self.config, "use_unsloth", False) and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA (Unsloth enabled)")
            return device

        if self.config.use_mps and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders)")
            # Optimize MPS settings for better memory management
            if hasattr(torch.backends.mps, 'set_per_process_memory_fraction'):
                # Reduce memory fraction to prevent OOM (0.7 = 70% of available memory)
                # This leaves room for system and other allocations
                torch.backends.mps.set_per_process_memory_fraction(0.7)
                logger.info("MPS memory fraction set to 0.7 (70%) to prevent OOM")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device
    
    def _load_mlx_model_for_generation(self, model_path: str):
        """Load MLX model for faster generation (similar to preload_model.py)"""
        try:
            from mlx_lm import load
            logger.info(f"Loading MLX model for faster generation: {model_path}")
            
            # Auto-detect quantization from path name if not in config
            quantize_bits = None
            if self.config.mlx_quantization:
                if self.config.mlx_quantization == "q4_bit":
                    quantize_bits = 4
                elif self.config.mlx_quantization == "q8_bit":
                    quantize_bits = 8
            elif "/q4" in model_path.lower() or "q4_bit" in model_path.lower():
                quantize_bits = 4
                logger.info("Auto-detected Q4 quantization from path name")
            elif "/q8" in model_path.lower() or "q8_bit" in model_path.lower():
                quantize_bits = 8
                logger.info("Auto-detected Q8 quantization from path name")
            
            if os.path.exists(model_path):
                # If model path contains q4/q8, the model is already quantized
                # Don't pass quantize parameter - just load the pre-quantized model
                model_already_quantized = "/q4" in model_path.lower() or "/q8" in model_path.lower()
                
                if model_already_quantized:
                    logger.info(f"Loading pre-quantized MLX model from {model_path}...")
                    logger.info("  (Model is already quantized - no need to pass quantize parameter)")
                    self.mlx_model, self.mlx_tokenizer = load(model_path)
                    # Extract quantization level from path for logging
                    if "/q4" in model_path.lower():
                        quantize_bits = 4
                    elif "/q8" in model_path.lower():
                        quantize_bits = 8
                    logger.info(f"✓ Loaded pre-quantized model ({quantize_bits}-bit)")
                elif quantize_bits:
                    # Model path doesn't indicate quantization, but we want to quantize on load
                    logger.info(f"Loading MLX model and applying {quantize_bits}-bit quantization...")
                    try:
                        self.mlx_model, self.mlx_tokenizer = load(model_path, quantize=quantize_bits)
                        logger.info(f"✓ Loaded with {quantize_bits}-bit quantization")
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Quantization parameter not supported in this MLX version: {e}")
                        logger.info("Loading model without quantization parameter (model will be full precision)")
                        self.mlx_model, self.mlx_tokenizer = load(model_path)
                        quantize_bits = None  # Reset since quantization failed
                else:
                    logger.info("Loading MLX model without quantization (full precision)...")
                    self.mlx_model, self.mlx_tokenizer = load(model_path)
                
                logger.info("✓ MLX model loaded for generation (5-10x faster than PyTorch MPS)")
                if quantize_bits:
                    logger.info(f"  Using {quantize_bits}-bit quantization for faster inference")
                else:
                    logger.info("  Using full precision (this is MUCH slower). For best throughput, use q4/q8:")
                    logger.info(f"    uv run python scripts/utils/convert_to_mlx.py --hf-path {self.config.base_model} --mlx-path ./mlx_model/q4 --quantize q4_bit")

                # Warm up MLX generation to pay compilation/Metal setup once (not inside epoch timing).
                # This is a major source of “Batch 0 took minutes” behavior.
                try:
                    from mlx_lm import generate as mlx_generate
                    warmup_prompt = "Write high-quality python code:\n\nReturn 1+1.\n\nCode:"
                    warmup_start = time.time()
                    _ = mlx_generate(
                        self.mlx_model,
                        self.mlx_tokenizer,
                        prompt=warmup_prompt,
                        max_tokens=4,
                        sampler=None,
                    )
                    warmup_time = time.time() - warmup_start
                    logger.info(f"✓ MLX warmup complete (took {warmup_time:.2f}s). Subsequent batches should be faster.")
                    self.generation_warmup_done = True
                except Exception as e:
                    logger.debug(f"MLX warmup failed (non-critical): {e}")
                
                # Verify tokenizer compatibility between MLX and PyTorch
                # This is CRITICAL: if tokenizers produce different token IDs, we'll get NaN logits
                logger.info("Verifying tokenizer compatibility between MLX and PyTorch...")
                test_texts = [
                    "Hello, world!",
                    "def function():",
                    "# Python comment",
                    "int main() {",
                    "print('test')",
                    "Write high-quality python code:"
                ]
                mismatches = []
                for test_text in test_texts:
                    try:
                        # Use encode with same parameters
                        mlx_tokens = self.mlx_tokenizer.encode(test_text, add_special_tokens=False)
                        pytorch_tokens = self.tokenizer.encode(test_text, add_special_tokens=False)
                        if mlx_tokens != pytorch_tokens:
                            mismatches.append((test_text, mlx_tokens, pytorch_tokens))
                    except Exception as e:
                        logger.warning(f"Error testing tokenizer compatibility for '{test_text}': {e}")
                
                if mismatches:
                    logger.error("⚠️  CRITICAL: Tokenizer mismatch detected between MLX and PyTorch!")
                    logger.error("  This WILL cause invalid token IDs and NaN logits during training!")
                    logger.error(f"  Found {len(mismatches)} mismatches out of {len(test_texts)} test cases")
                    for test_text, mlx_tokens, pytorch_tokens in mismatches[:5]:  # Show first 5 mismatches
                        logger.error(f"  Text: '{test_text}'")
                        logger.error(f"    MLX tokens:     {mlx_tokens}")
                        logger.error(f"    PyTorch tokens: {pytorch_tokens}")
                        logger.error(f"    Match: {mlx_tokens == pytorch_tokens}")
                    logger.error("  SOLUTION: Ensure both tokenizers are from the same model.")
                    logger.error("  The MLX model and PyTorch model should use identical tokenizers.")
                    logger.error("  This mismatch is likely causing the NaN logits issue!")
                    logger.error("  ACTION: Consider using PyTorch generation instead of MLX, or")
                    logger.error("         ensure MLX model was converted from the same base model.")
                else:
                    logger.info("✓ Tokenizer compatibility verified - MLX and PyTorch tokenizers match")
            elif model_path:
                # User specified MLX path but it doesn't exist
                logger.warning(f"MLX model path specified but not found: {model_path}")
                logger.info("Will use PyTorch for generation (slower)")
                logger.info("Tip: Convert model to MLX format:")
                logger.info(f"  uv run python scripts/utils/convert_to_mlx.py --hf-path {self.config.base_model} --mlx-path {model_path}")
                if self.config.mlx_quantization:
                    logger.info(f"  --quantize {self.config.mlx_quantization}")
            else:
                # No MLX model specified or found
                logger.info("MLX model not found. Will use PyTorch for generation (slower).")
                logger.info("Tip: Convert model to MLX format for 5-10x faster generation:")
                logger.info(f"  uv run python scripts/utils/convert_to_mlx.py --hf-path {self.config.base_model} --mlx-path ./mlx_model/q8 --quantize q8_bit")
                logger.info("Then update config.yaml:")
                logger.info("  hardware:")
                logger.info("    use_mlx_for_generation: true")
                logger.info("    mlx_model_path: ./mlx_model/q8")
        except ImportError:
            logger.warning("MLX not available. Install with: uv pip install mlx mlx-lm")
            logger.info("Using PyTorch MPS for generation (slower)")
        except Exception as e:
            logger.warning(f"Could not load MLX model: {e}")
            logger.info("Falling back to PyTorch MPS for generation")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _generate_prompt_variation(self, base_prompt: str, language: str, sample_idx: int, num_samples: int, epoch: int = 0, nonce: int = 0) -> str:
        """Generate a prompt variation to increase code diversity
        
        Args:
            base_prompt: The original prompt
            language: Programming language
            sample_idx: Index of this sample (0 to num_samples-1)
            num_samples: Total number of samples per prompt
            
        Returns:
            Varied prompt string
        """
        import random
        import hashlib

        # Deterministic per-epoch/per-sample randomness so each epoch naturally gets different prompt variants
        seed_material = f"{epoch}:{nonce}:{sample_idx}:{num_samples}:{language}:{base_prompt}".encode("utf-8")
        seed = int(hashlib.md5(seed_material).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Different prompt templates for diversity (expanded for more variety)
        templates = [
            f"Write high-quality {language} code:\n\n{{prompt}}\n\nCode:",
            f"Implement the following in {language}:\n\n{{prompt}}\n\nSolution:",
            f"Create {language} code for:\n\n{{prompt}}\n\nCode:",
            f"Write efficient {language} code:\n\n{{prompt}}\n\nImplementation:",
            f"Generate {language} code:\n\n{{prompt}}\n\nCode:",
            f"Write clean {language} code:\n\n{{prompt}}\n\nSolution:",
            f"Develop a {language} solution:\n\n{{prompt}}\n\nCode:",
            f"Build {language} code:\n\n{{prompt}}\n\nImplementation:",
            f"Write well-structured {language} code:\n\n{{prompt}}\n\nCode:",
            f"Create optimized {language} code:\n\n{{prompt}}\n\nSolution:",
        ]
        
        # Use different templates for different samples (deterministic randomization)
        template_idx = (sample_idx + rng.randint(0, len(templates) - 1)) % len(templates)
        template = templates[template_idx]
        
        # Add more diverse variations to the prompt text itself
        prompt_variations = [
            base_prompt,
            base_prompt + " Make sure the code is efficient.",
            base_prompt + " Include proper error handling.",
            base_prompt + " Add comments for clarity.",
            base_prompt + " Optimize for performance.",
            base_prompt + " Use best practices.",
            base_prompt + " Make it readable and maintainable.",
            base_prompt + " Consider edge cases.",
            base_prompt + " Write production-ready code.",
            base_prompt + " Focus on code quality.",
        ]
        
        # Epoch-level style directives to force diversity even if decoding is low-entropy
        style_directives = [
            "Prefer an iterative approach (avoid recursion if possible).",
            "Prefer a recursive approach if it makes the solution clearer.",
            "Prefer a minimal, concise solution (but still correct).",
            "Prefer a well-documented solution with clear comments.",
            "Prefer a solution optimized for performance (avoid unnecessary allocations).",
            "Prefer a solution optimized for readability and maintainability.",
            "Prefer using standard library utilities idiomatically.",
            "Prefer explicit error handling and clear failure modes.",
            "Prefer functional style where appropriate.",
            "Prefer an alternative algorithmic approach if possible (not the most obvious one).",
        ]
        style_idx = (epoch + sample_idx + nonce) % len(style_directives)

        # Use different prompt variation based on sample index (deterministic randomization)
        variation_idx = (sample_idx + rng.randint(0, len(prompt_variations) - 1)) % len(prompt_variations)
        varied_prompt = prompt_variations[variation_idx]

        # Add an epoch “salt” so the same prompt yields different completions across epochs.
        epoch_salt = (
            f"\n\n[Epoch Variation: {epoch + 1} | nonce={nonce}]\n"
            f"- {style_directives[style_idx]}\n"
            f"- IMPORTANT: Try to produce a meaningfully different solution than previous epochs "
            f"(different structure/approach), while staying correct.\n"
        )

        return template.format(prompt=varied_prompt + epoch_salt)
    
    def generate_student_samples(self, prompts: List[str], languages: List[str], num_samples: int = 4, epoch: int = 0) -> List[Dict]:
        """Generate multiple samples from student model for each prompt (optimized for M5)"""
        # Unsloth toggle (CUDA-only): switch model to inference-optimized mode before generation
        if self._unsloth_enabled and self._unsloth_flm is not None:
            try:
                self._unsloth_flm.for_inference(self.model)
            except Exception:
                pass

        # Use MLX for generation if available (much faster than PyTorch MPS)
        if self.mlx_model is not None and self.mlx_tokenizer is not None:
            return self._generate_with_mlx(prompts, languages, num_samples, epoch=epoch)
        
        # Fall back to PyTorch MPS
        return self._generate_with_pytorch(prompts, languages, num_samples, epoch=epoch)
    
    def _generate_with_mlx(self, prompts: List[str], languages: List[str], num_samples: int, epoch: int = 0) -> List[Dict]:
        """Generate using MLX (5-10x faster than PyTorch MPS on Apple Silicon)
        
        Similar to preload_model.py, uses pre-compiled MLX model for fast generation.
        """
        from mlx_lm import generate as mlx_generate
        
        samples = []
        all_formatted_prompts = []
        # Keep sample_idx so we can force sampling on samples > 0 per prompt (improves uniqueness)
        prompt_metadata = []  # (prompt, language, sample_idx)
        
        for prompt, language in zip(prompts, languages):
            # Use prompt variations for diversity
            for sample_idx in range(num_samples):
                # IMPORTANT: use a per-sample nonce so prompts differ materially across samples.
                # This significantly reduces identical completions even when using greedy decoding.
                formatted_prompt = self._generate_prompt_variation(prompt, language, sample_idx, num_samples, epoch=epoch, nonce=sample_idx)
                all_formatted_prompts.append(formatted_prompt)
                prompt_metadata.append((prompt, language, sample_idx))
        
        # MLX generation is much faster - can process sequentially or in small batches
        # Similar to preload_model.py, use minimal parameters for fastest generation
        logger.debug(f"Generating {len(all_formatted_prompts)} samples with MLX (5-10x faster than PyTorch MPS)...")
        
        import hashlib

        # Import sampler once (avoid per-sample import overhead)
        try:
            from mlx_lm.sample import sample as mlx_sample
        except Exception:
            mlx_sample = None

        # Track duplicates within this generation call to avoid wasting samples.
        seen_hashes_in_call = set()

        for formatted_prompt, (prompt, language, sample_idx) in zip(all_formatted_prompts, prompt_metadata):
            try:
                # MLX generate is optimized for Apple Silicon
                # Use minimal parameters for fastest generation (same as preload_model.py)
                # MLX automatically optimizes for Apple Silicon
                # Throughput note:
                # Generation TPS is measured over *generated* tokens only, but MLX still pays a prefill cost
                # for the prompt. Too-small max_tokens makes TPS look artificially low. Use a larger cap to
                # amortize prompt prefill (still safe because training truncates to max_length anyway).
                max_gen_tokens = min(128, self.config.max_length // 4)
                
                # MLX performance note:
                # Passing a Python `sampler` callback is invoked once per token and can significantly reduce
                # throughput. For maximum tokens/sec, prefer greedy decoding (sampler=None).
                #
                # Strategy:
                # - Attempt 0: greedy (fast)
                # - Retry attempts (only if needed for cross-epoch novelty): enable sampling to diversify
                base_temp = self.config.generation_temperature
                base_top_k = self.config.top_k
                base_top_p = self.config.top_p
                
                # Cross-epoch novelty:
                # Retry a couple times with different prompt nonce if we reproduce code from a previous epoch.
                # IMPORTANT for throughput: retries multiply compute but we only count final tokens.
                # Keep retries minimal; only do cross-epoch retry (epoch > 0) and only when we collide
                # with previously-seen (global) code.
                # If we always run greedy for every sample, duplicates are common and we waste generation time.
                # Force sampling for samples beyond the first sample per prompt to improve uniqueness.
                # We want unique samples WITHOUT slowing down the common case.
                # Strategy:
                # - Attempt 0: greedy for ALL samples (fast, maximizes tokens/sec)
                # - If we detect a duplicate hash (call/epoch/global): retry with higher-entropy sampling + new nonce
                max_novelty_attempts = 3 if epoch > 0 else 2
                generated_text = ""
                generated_code = ""
                code_hash = ""
                used_prompt = formatted_prompt
                for attempt in range(max_novelty_attempts):
                    sampler = None
                    # Enable sampling ONLY on retries (duplicates detected). This avoids a Python callback per token
                    # on the common (unique) path and significantly improves overall throughput.
                    if attempt > 0 and mlx_sample is not None and base_temp and base_temp > 0:
                        import random
                        # Retry attempts should explore more to avoid duplicates.
                        # Use a higher base temperature and wider variation on retries.
                        temp_base = max(base_temp, 1.0)
                        if attempt > 0:
                            temp_base = max(temp_base, 1.25)
                        temp_variation = 0.6 if attempt > 0 else 0.4
                        sample_temp = temp_base + (random.random() - 0.5) * temp_variation
                        sample_temp = max(0.8, min(1.7, sample_temp))

                        # Also widen nucleus/top-k slightly on retries to increase diversity.
                        top_k = max(base_top_k, 50)
                        top_p = max(base_top_p, 0.95)
                        if attempt > 0:
                            top_k = max(top_k, 80)
                            top_p = max(top_p, 0.98)

                        sampler = lambda logits: mlx_sample(logits, temp=sample_temp, top_k=top_k, top_p=top_p)

                    # Attempt 0 uses the pre-built per-sample prompt (unique nonce already baked in).
                    # On retries, change the prompt nonce so the model sees a different “context salt”.
                    attempt_prompt = formatted_prompt if attempt == 0 else self._generate_prompt_variation(
                        prompt,
                        language,
                        sample_idx=sample_idx,
                        num_samples=num_samples,
                        epoch=epoch,
                        nonce=(sample_idx * 10 + attempt),
                    )
                    used_prompt = attempt_prompt
                    gen_call_start = time.time()
                    generated_text = mlx_generate(
                        self.mlx_model,
                        self.mlx_tokenizer,
                        prompt=attempt_prompt,
                        max_tokens=max_gen_tokens,  # Reduced for faster generation
                        sampler=sampler if sampler else None,  # Use sampler for temperature control
                    )
                    gen_call_seconds = time.time() - gen_call_start
                    
                    # Extract only the generated part (remove prompt)
                    if generated_text.startswith(attempt_prompt):
                        generated_code = generated_text[len(attempt_prompt):].strip()
                    else:
                        generated_code = generated_text.strip()
                    
                    normalized = ' '.join(generated_code.split())
                    code_hash = hashlib.md5(normalized.encode()).hexdigest()
                    # Retry only if we collided with something we've already seen:
                    # - within this generation call (most common)
                    # - earlier in the epoch
                    # - in previous epochs (global)
                    is_unique = (
                        (code_hash not in seen_hashes_in_call)
                        and (code_hash not in self._epoch_code_hashes)
                        and (code_hash not in self._global_code_hashes)
                    )
                    if is_unique:
                        break
                
                if code_hash:
                    self._epoch_code_hashes.add(code_hash)
                    seen_hashes_in_call.add(code_hash)
                
                # Optional micro-profiling (debug-only): log prompt vs output token sizes and per-sample time.
                # This helps identify whether we're prefill-bound (long prompts) or decode-bound.
                # NOTE: We avoid logging every sample at INFO to reduce overhead.
                if logger.isEnabledFor(logging.DEBUG):
                    try:
                        # Approximate tokens using the PyTorch tokenizer (available and fast)
                        prompt_tok = len(self.tokenizer.encode(used_prompt, add_special_tokens=False))
                        out_tok = len(self.tokenizer.encode(generated_code, add_special_tokens=False))
                        logger.debug(f"MLX gen sample: prompt_tokens={prompt_tok}, output_tokens={out_tok}, max_gen_tokens={max_gen_tokens}")
                    except Exception:
                        pass

                samples.append({
                    'prompt': prompt,
                    'language': language,
                    'code': generated_code,
                    # Keep the exact prompt used (may differ due to novelty retry) so training can tokenize later.
                    # This drastically reduces overhead during generation (we only tokenize selected samples for training).
                    'full_prompt': used_prompt,
                    # Per-sample timing / token stats (helps compute avg-per-sample tok/s)
                    'gen_seconds': float(gen_call_seconds) if 'gen_call_seconds' in locals() else 0.0,
                    'output_tokens': int(len(self.tokenizer.encode(generated_code, add_special_tokens=False))) if generated_code else 0,
                    'code_hash': code_hash,
                })
            except Exception as e:
                logger.warning(f"MLX generation failed for prompt: {e}")
                # Fall back to empty code
                formatted_prompt = f"Write high-quality {language} code:\n\n{prompt}\n\nCode:"
                samples.append({
                    'prompt': prompt,
                    'language': language,
                    'code': '',
                    'full_prompt': formatted_prompt,
                })
        
        return samples
    
    def _generate_with_pytorch(self, prompts: List[str], languages: List[str], num_samples: int, epoch: int = 0) -> List[Dict]:
        """Generate using PyTorch MPS (fallback if MLX not available)"""
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            # Batch process all prompts at once for efficiency
            all_formatted_prompts = []
            prompt_metadata = []
            
            for prompt, language in zip(prompts, languages):
                # Use prompt variations for diversity (same as MLX)
                for sample_idx in range(num_samples):
                    formatted_prompt = self._generate_prompt_variation(prompt, language, sample_idx, num_samples, epoch=epoch, nonce=sample_idx)
                    all_formatted_prompts.append(formatted_prompt)
                    prompt_metadata.append((prompt, language))
            
            # Batch tokenize
            if all_formatted_prompts:
                inputs = self.tokenizer(
                    all_formatted_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length,
                    padding=True
                )
                # Move to device with optimized MPS settings
                # Use non_blocking for faster transfer on unified memory
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                
                # Generate in smaller batches for M5 to avoid memory pressure
                # MPS benefits from smaller batches on unified memory
                # Further reduced batch size to prevent OOM
                batch_size = min(1, len(all_formatted_prompts))  # Reduced to 1 to prevent MPS OOM
                for i in range(0, len(all_formatted_prompts), batch_size):
                    batch_inputs = {
                        'input_ids': inputs['input_ids'][i:i+batch_size],
                        'attention_mask': inputs['attention_mask'][i:i+batch_size]
                    }
                    
                    # Optimized generation parameters for M5 MPS
                    # Reduced max_new_tokens to prevent OOM
                    # Use sampling for diversity in generated code
                    # Vary temperature significantly across samples for more diversity
                    import random
                    base_temp = self.config.generation_temperature
                    temp_variation = 0.4  # Increased from ±0.2 to ±0.4 for more diversity
                    sample_temp = base_temp + (random.random() - 0.5) * temp_variation
                    sample_temp = max(0.7, min(1.3, sample_temp))  # Clamp between 0.7 and 1.3 (wider range)
                    
                    generation_config = {
                        "max_new_tokens": min(128, self.config.max_length // 4),  # Reduced from 256 to save memory
                        "do_sample": True,  # Enable sampling to use temperature/top_p/top_k
                        "temperature": sample_temp,  # Vary temperature for diversity
                        "top_k": self.config.top_k,
                        "top_p": self.config.top_p,
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "num_return_sequences": 1,
                        "use_cache": True,  # Critical for MPS performance
                        "output_scores": False,
                        "return_dict_in_generate": False,
                        "repetition_penalty": 1.1 + random.random() * 0.2,  # Vary repetition penalty (1.1 to 1.3) for more diversity
                    }
                    
                    # Synchronize MPS before generation for better performance
                    if device.type == "mps":
                        torch.mps.synchronize()
                    
                    outputs = self.model.generate(**batch_inputs, **generation_config)
                    
                    # Synchronize after generation
                    if device.type == "mps":
                        torch.mps.synchronize()
                    
                    # Decode all at once
                    for j, output in enumerate(outputs):
                        idx = i + j
                        if idx < len(prompt_metadata):
                            prompt, language = prompt_metadata[idx]
                            input_len = batch_inputs['input_ids'][j].shape[0]
                            generated_text = self.tokenizer.decode(
                                output[input_len:],
                                skip_special_tokens=True
                            )
                            
                            samples.append({
                                'prompt': prompt,
                                'language': language,
                                'code': generated_text,
                                # Keep prompt used; tokenize later only for selected samples.
                                'full_prompt': all_formatted_prompts[i * batch_size + j],
                            })
        
        return samples
    
    def _get_teacher_code_cached(self, prompt: str, language: str) -> str:
        """Get teacher code with caching"""
        cache_key = f"{prompt}:{language}"
        if cache_key not in self.teacher_cache:
            self.teacher_cache[cache_key] = self.teacher.generate(prompt, language)
            # teacher reference generation call
            self.cache_stats['teacher_gen_calls'] += 1
            self.cache_stats['api_calls'] += 1  # legacy
            self.cache_stats['cache_misses'] += 1
        else:
            self.cache_stats['teacher_gen_cache_hits'] += 1
            self.cache_stats['cache_hits'] += 1
        return self.teacher_cache[cache_key]
    
    def _process_sample_reward(self, sample: Dict) -> Tuple[float, Dict]:
        """Process a single sample to compute reward (for parallel execution)"""
        try:
            # Get teacher's reference code (cached)
            teacher_code = self._get_teacher_code_cached(
                sample['prompt'],
                sample['language']
            )
            
            # Score student code with caching
            student_code_key = f"{sample['code']}:{sample['prompt']}:{sample['language']}"
            if student_code_key in self.teacher_score_cache:
                student_score = self.teacher_score_cache[student_code_key]
                self.cache_stats['teacher_score_cache_hits'] += 1
            else:
                self.cache_stats['teacher_score_calls'] += 1
                student_score = self.teacher.score_code(
                    sample['code'],
                    sample['prompt'],
                    sample['language'],
                    use_cache=True  # Use cache in score_code as well (defensive caching)
                )
                self.teacher_score_cache[student_code_key] = student_score
            
            # Score teacher code (baseline) - cache this with a special prefix to distinguish from student scores
            # Teacher code doesn't change, so it's safe to cache across epochs
            teacher_code_key = f"TEACHER_CODE:{teacher_code}:{sample['prompt']}:{sample['language']}"
            if teacher_code_key in self.teacher_score_cache:
                teacher_score = self.teacher_score_cache[teacher_code_key]
                self.cache_stats['teacher_score_cache_hits'] += 1
            else:
                self.cache_stats['teacher_score_calls'] += 1
                teacher_score = self.teacher.score_code(
                    teacher_code,
                    sample['prompt'],
                    sample['language'],
                    use_cache=True  # Use cache in score_code as well (defensive caching)
                )
                self.teacher_score_cache[teacher_code_key] = teacher_score
            
            # Normalized reward (relative to teacher)
            reward = student_score / (teacher_score + 1e-6)
            
            # Apply reward bonuses for specific improvements (if enabled)
            if self.config.reward_bonuses:
                reward = self._apply_reward_bonuses(reward, sample['code'], sample['language'], student_score)
            
            dataset_entry = {
                'prompt': sample['prompt'],
                'language': sample['language'],
                'student_code': sample['code'],
                'teacher_code': teacher_code,
                'student_score': float(student_score),
                'teacher_score': float(teacher_score),
                'reward': float(reward),
                'scoring_breakdown': {
                    'correctness': 0.3,
                    'code_quality': 0.3,
                    'efficiency': 0.2,
                    'documentation': 0.2
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return reward, dataset_entry
        except Exception as e:
            # Note: scoring_errors is incremented in compute_rewards when this exception is caught
            logger.warning(f"Error processing sample: {e}")
            return 0.5, None
    
    def _filter_duplicate_samples(self, samples: List[Dict]) -> List[Dict]:
        """Filter out duplicate samples to improve diversity
        
        Args:
            samples: List of generated samples
            
        Returns:
            Filtered list with duplicates removed (keeping first occurrence)
        """
        import hashlib
        seen_hashes = set()
        unique_samples = []
        
        for sample in samples:
            code = sample.get('code', '')
            if not code:
                continue
            
            # Normalize code (remove whitespace differences)
            code_hash = sample.get('code_hash')
            if not code_hash:
                normalized = ' '.join(code.split())
                code_hash = hashlib.md5(normalized.encode()).hexdigest()
                sample['code_hash'] = code_hash
            
            if code_hash not in seen_hashes:
                seen_hashes.add(code_hash)
                unique_samples.append(sample)
            else:
                logger.debug(f"Filtered duplicate sample (hash: {code_hash[:8]}...)")
        
        if len(unique_samples) < len(samples):
            logger.info(f"Filtered {len(samples) - len(unique_samples)} duplicate samples (diversity: {len(unique_samples)/len(samples)*100:.1f}%)")
        
        return unique_samples
    
    def _apply_reward_bonuses(self, base_reward: float, code: str, language: str, student_score: float) -> float:
        """Apply reward bonuses for specific improvements"""
        bonus = 0.0
        
        # Bonus for good documentation (has docstrings/comments)
        doc_keywords = {
            'python': ['"""', "'''", '# ', 'def ', 'class '],
            'cpp': ['//', '/*', '/**'],
            'rust': ['//', '///', '//!']
        }
        keywords = doc_keywords.get(language.lower(), ['//', '#'])
        has_docs = any(keyword in code for keyword in keywords)
        if has_docs:
            bonus += 0.05  # 5% bonus for documentation
        
        # Bonus for efficiency (has optimizations, uses appropriate data structures)
        efficiency_indicators = ['dict', 'set', 'hash', 'cache', 'memo', 'optimize']
        has_efficiency = any(indicator in code.lower() for indicator in efficiency_indicators)
        if has_efficiency:
            bonus += 0.03  # 3% bonus for efficiency
        
        # Bonus for code quality (proper structure, error handling)
        quality_indicators = ['try:', 'except', 'if __name__', 'assert', 'raise']
        has_quality = any(indicator in code for indicator in quality_indicators)
        if has_quality:
            bonus += 0.02  # 2% bonus for code quality
        
        # Cap total bonus at 15% to prevent over-rewarding
        total_bonus = min(bonus, 0.15)
        
        return base_reward * (1.0 + total_bonus)
    
    def _calculate_code_diversity(self, codes: List[str]) -> Dict[str, float]:
        """Calculate code diversity metrics
        
        Returns:
            Dict with 'unique_count', 'total_count', 'unique_ratio', 'avg_similarity'
        """
        if not codes or len(codes) == 0:
            return {
                'unique_count': 0,
                'total_count': 0,
                'unique_ratio': 0.0,
                'avg_similarity': 0.0
            }
        
        # Use hash-based deduplication for fast uniqueness check
        import hashlib
        code_hashes = set()
        for code in codes:
            # Normalize code (remove whitespace differences)
            normalized = ' '.join(code.split())
            code_hash = hashlib.md5(normalized.encode()).hexdigest()
            code_hashes.add(code_hash)
        
        unique_count = len(code_hashes)
        total_count = len(codes)
        unique_ratio = unique_count / total_count if total_count > 0 else 0.0
        
        # Calculate average similarity using improved token-based similarity
        # This is more accurate than character-level for code
        if total_count > 1:
            similarities = []
            # Sample more pairs for better accuracy
            sample_size = min(200, total_count)  # Increased from 100 to 200
            for i in range(sample_size):
                code1 = codes[i]
                # Compare with more codes (increased from 10 to 20)
                for j in range(i + 1, min(i + 20, total_count)):
                    code2 = codes[j]
                    # Use token-based similarity (more accurate for code)
                    # Split into tokens (words/identifiers) for better comparison
                    tokens1 = set(code1.split())
                    tokens2 = set(code2.split())
                    if len(tokens1) + len(tokens2) > 0:
                        # Jaccard similarity on tokens
                        similarity = len(tokens1 & tokens2) / len(tokens1 | tokens2)
                        similarities.append(similarity)
            avg_similarity = np.mean(similarities) if similarities else 0.0
        else:
            avg_similarity = 0.0
        
        return {
            'unique_count': unique_count,
            'total_count': total_count,
            'unique_ratio': unique_ratio,
            'avg_similarity': avg_similarity
        }
    
    def compute_rewards(self, samples: List[Dict], save_to_dataset: bool = True) -> Tuple[List[float], List[Dict]]:
        """Compute rewards using teacher model with parallel processing (optimized for M5)
        
        Optimizations:
        - Parallel API calls with ThreadPoolExecutor
        - Caching to avoid redundant API calls
        - Adaptive worker count based on sample count
        - Progress tracking with tqdm
        """
        # Early return if no samples
        if not samples or len(samples) == 0:
            logger.warning("compute_rewards called with empty samples list")
            return [], []
        
        rewards = []
        dataset_entries = []
        
        # Optimize worker count: use more workers for larger batches, but cap at 8
        # More workers = faster but may hit API rate limits
        max_workers = min(8, max(2, len(samples) // 2), os.cpu_count() or 4)
        
        # Use existing executor if available (reuse thread pool)
        # Otherwise create a new one for this batch
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(self._process_sample_reward, sample): sample
                for sample in samples
            }
            
            # Collect results as they complete (no tqdm in training loop to avoid clutter)
            # Only show progress for large batches
            progress_iter = as_completed(future_to_sample)
            if len(samples) > 10:
                progress_iter = tqdm(progress_iter, total=len(samples), desc="Computing rewards", leave=False)
            
            for future in progress_iter:
                try:
                    reward, dataset_entry = future.result(timeout=120)  # 2 minute timeout per sample
                    rewards.append(reward)
                    if save_to_dataset and dataset_entry:
                        dataset_entries.append(dataset_entry)
                except TimeoutError:
                    self.error_stats['teacher_scoring_errors'] += 1
                    self.error_stats['scoring_errors'] += 1
                    logger.warning("Reward computation timed out, using default reward")
                    rewards.append(0.5)  # Default reward on timeout
                except Exception as e:
                    self.error_stats['teacher_scoring_errors'] += 1
                    self.error_stats['scoring_errors'] += 1
                    logger.warning(f"Error getting reward result: {e}")
                    rewards.append(0.5)  # Default reward on error
        
        return rewards, dataset_entries
    
    def _apply_lora(self, config: RLAIFConfig):
        """Apply LoRA or QLoRA to the model for efficient fine-tuning"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
            
            # QLoRA requires preparing the model for k-bit training
            if config.use_qlora:
                logger.info("Preparing model for QLoRA (4-bit + LoRA)...")
                if not config.use_4bit:
                    logger.warning("QLoRA requires 4-bit quantization. Enabling 4-bit quantization...")
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Determine target modules (auto-detect if not specified)
            if config.lora_target_modules is None:
                # Auto-detect based on model architecture
                model_name_lower = config.base_model.lower()
                if "qwen" in model_name_lower or "llama" in model_name_lower:
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                elif "gpt" in model_name_lower:
                    target_modules = ["c_attn", "c_proj", "c_fc"]
                else:
                    # Default: try common attention module names
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                logger.info(f"Auto-detected LoRA target modules: {target_modules}")
            else:
                target_modules = config.lora_target_modules
            
            # Create LoRA configuration
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_percent = 100 * trainable_params / total_params
            
            logger.info(f"✓ LoRA applied successfully!")
            logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}% of total)")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
            
            if config.use_qlora:
                logger.info("  Using QLoRA (4-bit quantization + LoRA) for maximum efficiency")
            else:
                logger.info("  Using LoRA (full precision + LoRA) for efficient fine-tuning")
            
            return self.model
            
        except ImportError:
            logger.error("peft library not found. Install with: pip install peft")
            raise
        except Exception as e:
            logger.error(f"Error applying LoRA: {e}")
            logger.warning("Continuing without LoRA (full model fine-tuning)")
            return self.model
    
    def _apply_curriculum_learning(self, dataset: CodeDataset) -> CodeDataset:
        """Apply curriculum learning: sort prompts by difficulty (start easy, increase difficulty)"""
        # Simple heuristic: shorter prompts are easier, longer prompts are harder
        # In a more sophisticated implementation, we could use prompt complexity metrics
        
        # Sort by prompt length (shorter = easier)
        sorted_data = sorted(dataset.data, key=lambda x: len(x.get('prompt', '')))
        
        # Create new dataset with sorted data (don't load from file, just copy structure)
        curriculum_dataset = CodeDataset.__new__(CodeDataset)  # Create instance without calling __init__
        curriculum_dataset.tokenizer = self.tokenizer
        curriculum_dataset.max_length = dataset.max_length
        curriculum_dataset.data = sorted_data  # Set data directly without loading from file
        
        logger.info(f"Applied curriculum learning: sorted {len(sorted_data)} samples by difficulty (shortest first)")
        return curriculum_dataset
    
    def compute_kl_penalty(self, log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence penalty with NaN/inf protection"""
        # Check for NaN or Inf values
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            logger.warning("NaN/Inf detected in log_probs, using zero KL penalty")
            return torch.tensor(0.0, device=log_probs.device)
        if torch.isnan(ref_log_probs).any() or torch.isinf(ref_log_probs).any():
            logger.warning("NaN/Inf detected in ref_log_probs, using zero KL penalty")
            return torch.tensor(0.0, device=log_probs.device)
        
        # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log_P - log_Q))
        # Here we use: kl = log_probs - ref_log_probs (difference in log probabilities)
        kl = log_probs - ref_log_probs
        
        # Clamp to prevent extreme values
        kl = torch.clamp(kl, min=-10.0, max=10.0)
        
        # Check result for NaN/Inf
        kl_mean = kl.mean()
        if torch.isnan(kl_mean) or torch.isinf(kl_mean):
            logger.warning("NaN/Inf in KL penalty computation, using zero")
            return torch.tensor(0.0, device=log_probs.device)
        
        return self.config.kl_penalty * kl_mean
    
    def _create_training_batch_from_samples(self, samples: List[Dict], original_prompts: List[str]) -> Dict:
        """Create training batch from generated samples
        
        This ensures we use the full sequences (prompt + generated code) for training,
        not just the original prompts from the DataLoader.
        """
        # Group samples by original prompt string.
        # NOTE: We no longer assume a fixed number of samples per prompt because we may skip duplicates.
        samples_by_prompt: Dict[str, List[int]] = {}
        for idx, s in enumerate(samples):
            p = s.get('prompt')
            if p is None:
                continue
            samples_by_prompt.setdefault(p, []).append(idx)
        
        # Collect input_ids and attention_masks from samples
        batch_input_ids = []
        batch_attention_masks = []
        
        for i, prompt in enumerate(original_prompts):
            # Pick a sample for this prompt.
            # If rewards are available, prefer the highest-reward sample for a stronger learning signal.
            idxs = samples_by_prompt.get(prompt, [])
            sample = None
            if idxs:
                best_idx = idxs[0]
                if hasattr(self, "_latest_batch_rewards") and self._latest_batch_rewards is not None:
                    try:
                        best_idx = max(idxs, key=lambda j: float(self._latest_batch_rewards[j]) if j < len(self._latest_batch_rewards) else -1.0)
                    except Exception:
                        best_idx = idxs[0]
                sample = samples[best_idx]
            if sample is not None:
                if 'input_ids' in sample and 'attention_mask' in sample:
                    batch_input_ids.append(sample['input_ids'])
                    batch_attention_masks.append(sample['attention_mask'])
                else:
                    # Fast path: tokenize only the selected sample (prompt + generated code).
                    # We avoid tokenizing *all* generated samples during generation to keep MLX throughput high.
                    full_prompt = sample.get('full_prompt') or f"Write high-quality {sample.get('language', 'python')} code:\n\n{prompt}\n\nCode:"
                    full_sequence = (full_prompt + (sample.get('code', '') or '')).strip()
                    tokenized = self.tokenizer(
                        full_sequence,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.config.max_length,
                        padding='max_length',
                        return_attention_mask=True,
                    )
                    batch_input_ids.append(tokenized['input_ids'].squeeze())
                    batch_attention_masks.append(tokenized['attention_mask'].squeeze())
            else:
                # Fallback if sample not found
                logger.warning(f"Sample not found for prompt {i}, using fallback")
                formatted_prompt = f"Write high-quality python code:\n\n{prompt}\n\nCode:"
                tokenized = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length,
                    padding='max_length',
                    return_attention_mask=True,
                )
                batch_input_ids.append(tokenized['input_ids'].squeeze())
                batch_attention_masks.append(tokenized['attention_mask'].squeeze())
        
        # Stack into tensors
        input_ids = torch.stack(batch_input_ids)
        attention_mask = torch.stack(batch_attention_masks)
        
        # Ensure tensors are on the correct device and dtype
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Validate tensors
        # Check for invalid token IDs (NaN, Inf, or out of vocabulary range)
        # Get vocab_size from model config (more reliable)
        vocab_size = getattr(self.model.config, 'vocab_size', len(self.tokenizer))
        if vocab_size == 0 or vocab_size is None:
            vocab_size = len(self.tokenizer)
        
        logger.debug(f"Validating input_ids: shape={input_ids.shape}, vocab_size={vocab_size}, min={input_ids.min().item()}, max={input_ids.max().item()}")
        
        if torch.isnan(input_ids.float()).any() or torch.isinf(input_ids.float()).any():
            logger.error("NaN/Inf detected in input_ids from samples!")
            # Replace with valid token IDs (pad token)
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            if pad_token_id is None:
                pad_token_id = 0
            input_ids = torch.where(torch.isnan(input_ids.float()) | torch.isinf(input_ids.float()), 
                                   torch.tensor(pad_token_id, device=input_ids.device, dtype=input_ids.dtype), 
                                   input_ids)
        
        # Check for out-of-vocabulary token IDs
        invalid_mask = (input_ids < 0) | (input_ids >= vocab_size)
        if invalid_mask.any():
            invalid_count = invalid_mask.sum().item()
            logger.error(f"Invalid token IDs detected: {invalid_count} tokens out of vocabulary range [0, {vocab_size})")
            logger.error(f"Token ID stats: min={input_ids.min().item()}, max={input_ids.max().item()}, vocab_size={vocab_size}")
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            if pad_token_id is None:
                pad_token_id = 0
            input_ids = torch.where(invalid_mask,
                                   torch.tensor(pad_token_id, device=input_ids.device, dtype=input_ids.dtype),
                                   input_ids)
        
        # Validate attention_mask
        if torch.isnan(attention_mask.float()).any() or torch.isinf(attention_mask.float()).any():
            logger.error("NaN/Inf detected in attention_mask from samples!")
            attention_mask = torch.where(torch.isnan(attention_mask.float()) | torch.isinf(attention_mask.float()),
                                       torch.tensor(0, device=attention_mask.device, dtype=attention_mask.dtype),
                                       attention_mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'prompt': original_prompts,
        }
    
    def train_step(self, batch: Dict, rewards: List[float]) -> Dict[str, float]:
        """Perform one training step with RLAIF (optimized for M5)"""
        # Unsloth toggle (CUDA-only): switch model to training-optimized mode before forward/backward
        if self._unsloth_enabled and self._unsloth_flm is not None:
            try:
                self._unsloth_flm.for_training(self.model)
            except Exception:
                pass

        self.model.train()
        
        # Initialize variables to avoid "referenced before assignment" errors
        ref_outputs = None
        ref_log_probs = None
        
        # Move to device with non_blocking for faster transfer
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
        
        # Enable gradient checkpointing to save memory (trades compute for memory)
        # This is especially important for MPS to prevent OOM
        # Note: use_cache must be False when gradient checkpointing is enabled
        gradient_checkpointing_enabled = False
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            gradient_checkpointing_enabled = True
            # Disable use_cache when gradient checkpointing is enabled to avoid warnings
            if hasattr(self.model, 'config'):
                self.model.config.use_cache = False
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False  # Explicitly disable cache when gradient checkpointing is enabled
        )
        
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Get reference log probs (from base model, frozen)
        # For efficiency, we use the model in eval mode as reference
        # In a true RLAIF setup, you'd have a separate frozen base model
        # Disable gradient checkpointing for reference pass (no gradients needed)
        original_use_cache = None
        if gradient_checkpointing_enabled:
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()
            # Restore use_cache for reference pass (no gradients, so cache is safe)
            if hasattr(self.model, 'config'):
                original_use_cache = getattr(self.model.config, 'use_cache', None)
                self.model.config.use_cache = True
        
        # Clear cache before reference pass to free memory
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        with torch.no_grad():
            self.model.eval()
            ref_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True  # Can use cache for reference pass (no gradients)
            )
            ref_log_probs = torch.nn.functional.log_softmax(ref_outputs.logits, dim=-1)
            self.model.train()  # Switch back to train mode
            
            # Delete reference outputs immediately to free memory
            del ref_outputs
        
        # Re-enable gradient checkpointing for training
        if gradient_checkpointing_enabled:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            # Restore use_cache=False for training with gradient checkpointing
            if hasattr(self.model, 'config'):
                self.model.config.use_cache = False
        
        # Clear cache after reference pass
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Validate inputs for NaN/Inf before computation
        # First check input_ids for invalid values
        vocab_size = getattr(self.model.config, 'vocab_size', len(self.tokenizer))
        if vocab_size == 0 or vocab_size is None:
            vocab_size = len(self.tokenizer)
        
        # Check for invalid token IDs BEFORE forward pass
        invalid_token_mask = (input_ids < 0) | (input_ids >= vocab_size)
        if invalid_token_mask.any():
            invalid_count = invalid_token_mask.sum().item()
            logger.error(f"Invalid token IDs detected BEFORE forward pass: {invalid_count} tokens out of range [0, {vocab_size})")
            logger.error(f"Token ID stats: min={input_ids.min().item()}, max={input_ids.max().item()}, vocab_size={vocab_size}")
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            if pad_token_id is None:
                pad_token_id = 0
            input_ids = torch.where(invalid_token_mask,
                                   torch.tensor(pad_token_id, device=input_ids.device, dtype=input_ids.dtype),
                                   input_ids)
        
        if torch.isnan(input_ids.float()).any() or torch.isinf(input_ids.float()).any():
            logger.error("NaN/Inf detected in input_ids! This will cause NaN logits.")
            logger.error(f"Input IDs stats: min={input_ids.min().item()}, max={input_ids.max().item()}, mean={input_ids.float().mean().item():.4f}")
            # Replace with pad token ID
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            if pad_token_id is None:
                pad_token_id = 0
            input_ids = torch.where(torch.isnan(input_ids.float()) | torch.isinf(input_ids.float()),
                                   torch.tensor(pad_token_id, device=input_ids.device, dtype=input_ids.dtype),
                                   input_ids)
        
        # Validate attention_mask
        if torch.isnan(attention_mask.float()).any() or torch.isinf(attention_mask.float()).any():
            logger.error("NaN/Inf detected in attention_mask! Replacing with zeros.")
            attention_mask = torch.where(torch.isnan(attention_mask.float()) | torch.isinf(attention_mask.float()),
                                       torch.tensor(0, device=attention_mask.device, dtype=attention_mask.dtype),
                                       attention_mask)
        
        # Check model weights for NaN/Inf BEFORE forward pass
        model_has_nan = False
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.error(f"NaN/Inf detected in model parameter BEFORE forward pass: {name}")
                logger.error(f"  Parameter shape: {param.shape}, dtype: {param.dtype}")
                model_has_nan = True
        
        if model_has_nan:
            logger.error("Model has NaN/Inf in weights! This will cause NaN logits.")
            logger.error("This may indicate corrupted model weights or numerical instability.")
            logger.error("Try reloading the model or using a different precision (float32 instead of bfloat16).")
        
        # Check embedding layer specifically (first layer that processes input_ids)
        try:
            embedding_layer = self.model.get_input_embeddings()
            if embedding_layer is not None:
                # Check if embedding weights have NaN
                if hasattr(embedding_layer, 'weight'):
                    emb_weight = embedding_layer.weight
                    if torch.isnan(emb_weight).any() or torch.isinf(emb_weight).any():
                        logger.error(f"NaN/Inf detected in embedding layer weights!")
                        logger.error(f"  Embedding shape: {emb_weight.shape}, dtype: {emb_weight.dtype}")
                        logger.error(f"  NaN count: {torch.isnan(emb_weight).sum().item()}, Inf count: {torch.isinf(emb_weight).sum().item()}")
        except Exception as e:
            logger.debug(f"Could not check embedding layer: {e}")
        
        # Forward pass with validated inputs
        # MPS doesn't need autocast - it handles bfloat16 natively
        # Using CUDA autocast on MPS causes issues and deprecation warnings
        device_type = next(self.model.parameters()).device.type
        if device_type == "mps":
            # MPS handles bfloat16 natively, no autocast needed
            # However, quantized models on MPS may have numerical issues
            # Check if model is quantized (BitsAndBytes)
            is_quantized = hasattr(self.model, 'hf_quantizer') or any(
                hasattr(module, 'weight') and hasattr(module.weight, 'SCB') 
                for module in self.model.modules()
            )
            
            if is_quantized:
                logger.debug("Using quantized model on MPS - monitoring for numerical stability")
                # Ensure inputs are on correct device
                if input_ids.device.type != "mps":
                    input_ids = input_ids.to("mps")
                if attention_mask.device.type != "mps":
                    attention_mask = attention_mask.to("mps")
            
            # Direct forward pass is more stable on MPS
            # For quantized models, we may need to handle dtype conversion
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False  # Explicitly disable cache when gradient checkpointing is enabled
                )
            except RuntimeError as e:
                if "NaN" in str(e) or "nan" in str(e).lower():
                    logger.error(f"Runtime error with NaN during forward pass: {e}")
                    logger.error("This is likely due to 4-bit quantization incompatibility with MPS.")
                    logger.error("SOLUTION: Disable quantization or use float32 instead of bfloat16")
                    raise
                raise
        elif device_type == "cuda":
            # Use CUDA autocast for CUDA devices
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.model.dtype == torch.bfloat16 else None):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False
                )
        else:
            # CPU or other devices - no autocast needed
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False
            )
        
        logits = outputs.logits
        
        # Check logits immediately after forward pass
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.error("NaN/Inf detected in logits AFTER forward pass! This will cause NaN loss.")
            logger.error(f"Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
            logger.error(f"Input IDs shape: {input_ids.shape}, min={input_ids.min().item()}, max={input_ids.max().item()}, vocab_size={vocab_size}")
            logger.error(f"Attention mask shape: {attention_mask.shape}, sum={attention_mask.sum().item()}")
            device = next(self.model.parameters()).device
            logger.error(f"Model dtype: {self.model.dtype}, device: {device}")
            logger.error(f"Device type: {device.type} (mps:0 means MPS device 0 - MPS IS being used)")
            
            # Check if this is a numerical stability issue with quantization on MPS
            is_quantized = hasattr(self.model, 'hf_quantizer') or any(
                hasattr(module, 'weight') and hasattr(module.weight, 'SCB') 
                for module in self.model.modules()
            )
            
            if device.type == "mps":
                if is_quantized:
                    logger.error("⚠️  CRITICAL: 4-bit quantization on MPS is causing NaN logits!")
                    logger.error("  BitsAndBytes quantization has known compatibility issues with MPS.")
                    logger.error("  SOLUTION 1 (Recommended): Disable quantization:")
                    logger.error("    In config.yaml, set model.use_4bit: false")
                    logger.error("  SOLUTION 2: Use float32 compute dtype (already attempted if enabled):")
                    logger.error("    The code should use float32 compute dtype for quantized models on MPS")
                    logger.error("  SOLUTION 3: Use CPU instead of MPS for quantized models")
                elif self.model.dtype == torch.bfloat16:
                    logger.error("⚠️  Potential bfloat16 numerical instability on MPS!")
                    logger.error("  MPS bfloat16 support may have issues with certain operations.")
                    logger.error("  SOLUTION: Try using float32 instead of bfloat16:")
                    logger.error("    In config.yaml, set model.use_4bit: false and use dtype: float32")
            
            # Check specific token IDs that might be problematic
            # Log a sample of input_ids to see if there's a pattern
            sample_input_ids = input_ids[0, :50].cpu().tolist()  # First 50 tokens of first batch
            logger.error(f"Sample input_ids (first 50 tokens): {sample_input_ids}")
            
            # Check if there are any special tokens that might cause issues
            special_token_ids = []
            if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
                special_token_ids.append(('bos', self.tokenizer.bos_token_id))
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                special_token_ids.append(('eos', self.tokenizer.eos_token_id))
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                special_token_ids.append(('pad', self.tokenizer.pad_token_id))
            if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
                special_token_ids.append(('unk', self.tokenizer.unk_token_id))
            logger.error(f"Special token IDs: {special_token_ids}")
            
            # Check if model weights have NaN
            nan_params = []
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    nan_count = torch.isnan(param).sum().item() + torch.isinf(param).sum().item()
                    nan_params.append((name, nan_count, param.shape))
                    logger.error(f"NaN/Inf detected in model parameter AFTER forward pass: {name} ({nan_count} values)")
            
            if nan_params:
                logger.error(f"Total parameters with NaN/Inf: {len(nan_params)}")
                logger.error("This suggests the model weights became corrupted during training or loading.")
            
            # Replace NaN/Inf with zeros as fallback
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Validate rewards
        rewards_array = np.array(rewards)
        if np.isnan(rewards_array).any() or np.isinf(rewards_array).any():
            logger.warning("NaN/Inf detected in rewards, replacing with 0.5")
            rewards = [0.5 if (np.isnan(r) or np.isinf(r)) else r for r in rewards]
            rewards_array = np.array(rewards)
        
        # Clamp rewards to reasonable range
        rewards = [max(0.0, min(1.0, float(r))) for r in rewards]
        
        # Compute policy gradient loss
        # Select log probs for generated tokens
        # Simplified: use average log prob as proxy
        selected_log_probs = log_probs[:, :-1, :].gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Validate selected_log_probs
        if torch.isnan(selected_log_probs).any() or torch.isinf(selected_log_probs).any():
            logger.warning("NaN/Inf in selected_log_probs, replacing with zeros")
            selected_log_probs = torch.nan_to_num(selected_log_probs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert rewards to tensor with validation
        reward_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        if len(reward_tensor.shape) == 1:
            reward_tensor = reward_tensor.unsqueeze(1)
        
        # Expand to match selected_log_probs shape
        if reward_tensor.shape[0] != selected_log_probs.shape[0]:
            logger.warning(f"Reward tensor shape mismatch: {reward_tensor.shape} vs {selected_log_probs.shape}")
            # Pad or truncate rewards to match
            if reward_tensor.shape[0] < selected_log_probs.shape[0]:
                # Pad with mean reward
                mean_reward = reward_tensor.mean().item()
                padding = torch.full((selected_log_probs.shape[0] - reward_tensor.shape[0], 1), mean_reward, device=self.device)
                reward_tensor = torch.cat([reward_tensor, padding], dim=0)
            else:
                reward_tensor = reward_tensor[:selected_log_probs.shape[0]]
        
        # Expand reward tensor to match sequence length
        if reward_tensor.shape[1] == 1 and len(selected_log_probs.shape) > 1:
            reward_tensor = reward_tensor.expand(-1, selected_log_probs.shape[1])
        
        # Ensure attention_mask is valid
        attn_mask = attention_mask[:, 1:]
        if attn_mask.shape != selected_log_probs.shape:
            logger.warning(f"Attention mask shape mismatch: {attn_mask.shape} vs {selected_log_probs.shape}")
            # Adjust attention mask
            if attn_mask.shape[1] > selected_log_probs.shape[1]:
                attn_mask = attn_mask[:, :selected_log_probs.shape[1]]
            elif attn_mask.shape[1] < selected_log_probs.shape[1]:
                padding = torch.zeros(attn_mask.shape[0], selected_log_probs.shape[1] - attn_mask.shape[1], device=self.device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([attn_mask, padding], dim=1)
        
        # Policy gradient: maximize log_prob * reward
        policy_loss = -(selected_log_probs * reward_tensor * attn_mask).mean()
        
        # Validate policy loss
        if torch.isnan(policy_loss) or torch.isinf(policy_loss):
            logger.warning("NaN/Inf in policy_loss, using zero")
            policy_loss = torch.tensor(0.0, device=self.device)
        
        # KL penalty
        ref_selected_log_probs = ref_log_probs[:, :-1, :].gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Validate ref_selected_log_probs
        if torch.isnan(ref_selected_log_probs).any() or torch.isinf(ref_selected_log_probs).any():
            logger.warning("NaN/Inf in ref_selected_log_probs, replacing with zeros")
            ref_selected_log_probs = torch.nan_to_num(ref_selected_log_probs, nan=0.0, posinf=0.0, neginf=0.0)
        
        kl_penalty = self.compute_kl_penalty(selected_log_probs, ref_selected_log_probs)
        
        # Total loss with validation
        total_loss = policy_loss + kl_penalty
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error("NaN/Inf in total_loss! Using zero loss as fallback.")
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Backward pass with timing
        # Scale loss by 1/gradient_accumulation_steps for proper gradient accumulation
        # This ensures gradients are averaged across accumulation steps
        scaled_loss = total_loss / self.config.gradient_accumulation_steps
        backprop_start = time.time()
        scaled_loss.backward()  # Accumulate gradients (don't zero grad here)
        backprop_time = time.time() - backprop_start
        
        # Calculate backprop tokens/sec (tokens processed during backward pass)
        # This is the number of tokens in the input sequence
        num_tokens = input_ids.numel()  # Total tokens in batch
        backprop_tokens_per_sec = num_tokens / backprop_time if backprop_time > 0 else 0
        self.training_metrics['backprop_tokens_per_sec'].append(backprop_tokens_per_sec)
        
        # Log to TensorBoard (will be logged in _log_stats if step matches logging_steps)
        
        # Clear intermediate tensors to free memory (optimization for M5)
        # Delete in order to free memory immediately
        # Note: ref_outputs was already deleted earlier (line 1535) to free memory immediately
        del logits, log_probs, ref_log_probs, selected_log_probs, ref_selected_log_probs, outputs
        
        # Calculate average reward safely (handle empty list)
        avg_reward = np.mean(rewards) if rewards and len(rewards) > 0 else 0.0
        
        return {
            'loss': total_loss.item(),  # Return unscaled loss for logging
            'policy_loss': policy_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'avg_reward': avg_reward,
        }
    
    def train(self, train_dataset: CodeDataset, eval_dataset: Optional[CodeDataset] = None):
        """Main training loop"""
        logger.info("Starting RLAIF training...")
        
        # Start system monitoring
        self._start_monitoring()
        
        # Apply curriculum learning if enabled (sort by difficulty)
        if self.config.curriculum_learning:
            train_dataset = self._apply_curriculum_learning(train_dataset)
        
        # Optimize DataLoader for M5: use num_workers=0 to avoid fork issues
        # M5 has unified memory, so single process is actually faster
        num_workers = 0 if self.config.use_mps else min(2, os.cpu_count() or 1)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=not self.config.curriculum_learning,  # Don't shuffle if using curriculum learning
            num_workers=num_workers,
            pin_memory=False,  # M5 doesn't benefit from pin_memory
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,  # Prefetch for faster data loading
            drop_last=False  # Keep all batches
        )

        # Compute baseline reward once (pre-training) for "gain vs baseline" checkpoint tagging
        if self.baseline_reward is None and int(getattr(self.config, "baseline_eval_batches", 0) or 0) > 0:
            try:
                self.baseline_reward = self._compute_baseline_reward(train_loader)
                # Persist baseline to output dir for offline reference
                try:
                    out_dir = Path(self.config.output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    with open(out_dir / "baseline_reward.json", "w", encoding="utf-8") as f:
                        json.dump({"baseline_reward": float(self.baseline_reward), "ts_iso": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Failed to compute baseline reward: {e}")
                self.baseline_reward = 0.0
        
        # Log training configuration
        logger.info(f"Training configuration:")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"  Samples per prompt: {self.config.num_samples_per_prompt}")
        logger.info(f"  Using MLX for generation: {self.mlx_model is not None}")
        if self.mlx_model is None:
            logger.warning("  ⚠️  MLX not enabled - generation will be slow. Enable for 5-10x speedup.")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_scheduler(
            self.config.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()  # Track epoch start time
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            self.stats['epoch'] = epoch + 1

            # Reset per-epoch diversity tracking; keep global hashes across epochs
            self._epoch_code_hashes = set()

            # Track gradient accumulation micro-steps (must be global within the epoch)
            # Otherwise we may never hit `gradient_accumulation_steps` on small datasets.
            micro_step_in_epoch = 0
            
            # CRITICAL: Clear student score cache at the start of each epoch
            # This ensures the model gets fresh feedback even if it generates similar code
            # The model should improve, so we need to re-score to see the improvement
            # Keep teacher code cache (that's fine to cache across epochs)
            student_cache_size_before = len(self.teacher_score_cache)
            # Remove only student code scores (keep teacher code scores)
            keys_to_remove = [k for k in self.teacher_score_cache.keys() if not k.startswith("TEACHER_CODE:")]
            for key in keys_to_remove:
                del self.teacher_score_cache[key]
            logger.info(f"Cleared {len(keys_to_remove)} student score cache entries at start of epoch {epoch + 1} (kept teacher code cache)")
            
            # Track API tokens at start of epoch
            epoch_start_api_tokens = self.training_metrics['api_tokens_sent']
            # Track teacher call/caching stats at start of epoch
            epoch_start_teacher_gen_calls = self.cache_stats.get('teacher_gen_calls', 0)
            epoch_start_teacher_gen_cache_hits = self.cache_stats.get('teacher_gen_cache_hits', 0)
            epoch_start_teacher_score_calls = self.cache_stats.get('teacher_score_calls', 0)
            epoch_start_teacher_score_cache_hits = self.cache_stats.get('teacher_score_cache_hits', 0)
            epoch_start_gen_errors = self.error_stats['generation_errors']
            epoch_start_scoring_errors = self.error_stats['scoring_errors']
            epoch_start_teacher_generate_errors = self.error_stats.get('teacher_generate_errors', 0)
            epoch_start_teacher_scoring_errors = self.error_stats.get('teacher_scoring_errors', 0)
            
            epoch_rewards = []
            epoch_best_reward_per_prompt = []  # per-batch: avg(max reward per prompt)
            epoch_losses = []
            epoch_generated_codes = []  # Track all generated code for diversity analysis
            
            # Track performance metrics per epoch
            epoch_gen_times = []  # Generation times
            epoch_reward_times = []  # Scoring times
            epoch_train_times = []  # Training times
            epoch_gen_tokens = []  # Tokens generated (output tokens from student model) AFTER dedup filtering (kept)
            epoch_gen_tokens_raw = []  # Tokens generated BEFORE dedup filtering (all sampled; true MLX throughput)
            epoch_gen_samples_raw = []  # Number of samples generated (raw, before dedup)
            epoch_gen_samples_kept = []  # Number of samples kept (after dedup)
            epoch_gen_sample_tps_raw = []  # Per-sample tokens/sec (raw samples)
            epoch_gen_sample_tps_kept = []  # Per-sample tokens/sec (kept samples)
            epoch_train_tokens = []  # Tokens used in training (input sequence tokens)
            epoch_reward_tokens = []  # API input tokens used for scoring
            epoch_reward_output_tokens = []  # API output tokens received from scoring
            
            # Batch dataset collection to reduce memory overhead
            dataset_batch = []
            dataset_batch_size = 10  # Collect 10 batches before extending main list
            
            # Initialize variables that might be used after loop (for error handling)
            batch_idx = -1
            rewards = []
            loss_dict = {'loss': 0.0, 'policy_loss': 0.0, 'kl_penalty': 0.0, 'avg_reward': 0.0}
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
                
                batch_start_time = time.time()
                
                # Initialize variables at start of batch to ensure they're always defined
                samples = []  # Initialize early to avoid "referenced before assignment" errors
                generation_error = False
                # Always assign a new list to rewards at the start of each batch
                # This ensures it's always defined even if we break early
                rewards = []  # New list for this batch (always assigned)
                dataset_entries = []
                reward_time = 0.001
                reward_api_tokens = 0
                reward_tokens_per_sec = 0
                train_time = 0.001
                train_num_tokens = 0
                train_tokens_per_sec = 0
                # Don't reassign loss_dict - it's already initialized before the loop
                # Just reset it for this batch
                loss_dict.update({'loss': 0.0, 'policy_loss': 0.0, 'kl_penalty': 0.0, 'avg_reward': 0.0})
                
                # Generate student samples
                prompts = batch['prompt']
                languages = batch['language']
                
                # Aggressive cache clearing to prevent MPS OOM
                # Clear cache every batch when using MPS to prevent memory buildup
                if torch.backends.mps.is_available():
                    # Aggressive cache clearing for MPS to prevent OOM
                    torch.mps.empty_cache()
                    import gc
                    gc.collect()  # Force garbage collection
                elif torch.cuda.is_available():
                    # For CUDA, clear less frequently
                    if batch_idx % 5 == 0:
                        torch.cuda.empty_cache()
                
                gen_start = time.time()
                # samples is already initialized above
                try:
                    samples_all = self.generate_student_samples(
                        prompts,
                        languages,
                        num_samples=self.config.num_samples_per_prompt,
                        epoch=epoch
                    )
                    # Compute raw generation throughput BEFORE dedup filtering.
                    # This reflects actual MLX decode speed (what you expect: ~7-9 tok/s on q4).
                    # Prefer already-computed token counts from generation (avoid extra tokenizer passes)
                    raw_num_tokens = sum(int(s.get('output_tokens', 0) or 0) for s in samples_all)
                    if raw_num_tokens == 0:
                        # Fallback for PyTorch samples that don't have output_tokens
                        for s in samples_all:
                            code = s.get('code', '')
                            if code:
                                raw_num_tokens += len(self.tokenizer.encode(code, add_special_tokens=False))
                    raw_tokens_per_sec = raw_num_tokens / max(gen_time := (time.time() - gen_start), 1e-6)

                    samples = samples_all
                    # Filter duplicates to improve diversity (this affects *kept* tokens/sec)
                    if len(samples) > 1:
                        before_n = len(samples)
                        samples = self._filter_duplicate_samples(samples)
                        after_n = len(samples)
                    else:
                        before_n = after_n = len(samples)

                    # Track per-sample TPS if available (MLX path provides gen_seconds/output_tokens)
                    def _collect_sample_tps(sample_list):
                        vals = []
                        for ss in sample_list or []:
                            ot = ss.get('output_tokens', 0) or 0
                            ts = ss.get('gen_seconds', 0.0) or 0.0
                            if ot > 0 and ts > 0:
                                vals.append(ot / ts)
                        return vals

                    epoch_gen_sample_tps_raw.extend(_collect_sample_tps(samples_all))
                    epoch_gen_sample_tps_kept.extend(_collect_sample_tps(samples))
                except Exception as e:
                    generation_error = True
                    self.error_stats['generation_errors'] += 1
                    logger.error(f"Error generating samples for batch {batch_idx}: {e}")
                    logger.warning(f"Skipping batch {batch_idx} due to generation error")
                    samples_all = []
                    raw_num_tokens = 0
                    raw_tokens_per_sec = 0.0
                    before_n = after_n = 0
                    samples = []  # Ensure samples is empty list on error
                gen_time = time.time() - gen_start
                
                # Synchronize MPS only if needed (after generation, before training)
                if torch.backends.mps.is_available() and batch_idx % 5 == 0:
                    torch.mps.synchronize()  # Sync periodically, not every batch
                
                # Track generation performance:
                # - raw_*: all generated samples (actual MLX throughput)
                # - kept_*: after dedup filtering (effective training throughput)
                num_tokens = sum(int(s.get('output_tokens', 0) or 0) for s in samples)
                if num_tokens == 0:
                    for s in samples:
                        code = s.get('code', '')
                        if code:
                            num_tokens += len(self.tokenizer.encode(code, add_special_tokens=False))
                tokens_per_sec = num_tokens / gen_time if gen_time > 0 else 0
                self.training_metrics['generation_tokens_per_sec'].append(tokens_per_sec)
                
                # Track epoch-level metrics
                # - gen_time reflects *all* sampled generation compute
                # - raw tokens reflect all sampled outputs
                # - kept tokens reflect post-dedup outputs
                if raw_num_tokens > 0:
                    epoch_gen_times.append(gen_time)
                    epoch_gen_tokens_raw.append(raw_num_tokens)
                    epoch_gen_tokens.append(num_tokens)
                    epoch_gen_samples_raw.append(before_n if 'before_n' in locals() else len(samples_all))
                    epoch_gen_samples_kept.append(after_n if 'after_n' in locals() else len(samples))
                
                # TensorBoard batch-level time series are handled later using a monotonic batch counter (`self._batch_step`).
                
                # Log generation performance (similar to preload_model.py)
                if batch_idx % 5 == 0:
                    batch_size_actual = len(batch['prompt'])
                    gen_time_str = f"{gen_time:.3f}s" if gen_time < 1.0 else f"{gen_time:.1f}s"
                    logger.info(
                        f"Batch {batch_idx} (size={batch_size_actual}) - Generation: {gen_time_str}, "
                        f"raw={raw_num_tokens:,} tokens ({raw_tokens_per_sec:.1f} tok/s), "
                        f"kept={num_tokens:,} tokens ({tokens_per_sec:.1f} tok/s), "
                        f"samples {before_n}→{after_n}"
                    )
                    
                    if self.mlx_model is not None:
                        # MLX generation
                        if tokens_per_sec > 1.0:
                            logger.info(f"✓ MLX generation: {tokens_per_sec:.1f} tokens/sec (excellent)")
                        elif tokens_per_sec > 0.5:
                            logger.info(f"✓ MLX generation: {tokens_per_sec:.1f} tokens/sec (good)")
                        else:
                            logger.warning(f"⚠️  MLX generation slower than expected: {tokens_per_sec:.1f} tokens/sec")
                            if self.config.mlx_quantization:
                                logger.info("  Consider using a different quantization level or no quantization")
                    elif tokens_per_sec < 1.0:
                        # PyTorch MPS is slow - provide actionable advice
                        logger.warning("⚠️  Slow generation. Consider using MLX for 5-10x speedup:")
                        logger.warning(f"  1. Convert model: uv run python scripts/utils/convert_to_mlx.py --hf-path {self.config.base_model} --mlx-path ./mlx_model/q8 --quantize q8_bit")
                        logger.warning("  2. Update config.yaml: hardware.use_mlx_for_generation: true")
                
                # Compute rewards and collect dataset entries (optimized)
                # Track API tokens before reward computation (both input and output)
                api_input_tokens_before = self.training_metrics['api_tokens_sent']
                api_output_tokens_before = self.training_metrics['api_tokens_received']
                # Track teacher call deltas for this batch
                teacher_gen_calls_before = int(self.cache_stats.get("teacher_gen_calls", 0))
                teacher_gen_hits_before = int(self.cache_stats.get("teacher_gen_cache_hits", 0))
                teacher_score_calls_before = int(self.cache_stats.get("teacher_score_calls", 0))
                teacher_score_hits_before = int(self.cache_stats.get("teacher_score_cache_hits", 0))
                reward_start = time.time()
                
                # Check if samples are empty before computing rewards
                if not samples or len(samples) == 0:
                    logger.warning(f"Batch {batch_idx}: No samples generated, skipping reward computation and training")
                    # All variables already initialized above, just skip computation
                    # rewards is already cleared above, just ensure dataset_entries is empty
                    dataset_entries = []
                else:
                    try:
                        rewards, dataset_entries = self.compute_rewards(samples, save_to_dataset=True)
                        # Ensure rewards is a list even if compute_rewards fails
                        if rewards is None:
                            rewards = []
                        if dataset_entries is None:
                            dataset_entries = []
                    except Exception as e:
                        logger.error(f"Error computing rewards for batch {batch_idx}: {e}")
                        rewards = []  # Assign empty list on error (always assigned)
                        dataset_entries = []
                    
                    reward_time = max(time.time() - reward_start, 0.001)  # Ensure non-zero (min 1ms for display)
                    # Track API tokens after reward computation (both input and output)
                    api_input_tokens_after = self.training_metrics['api_tokens_sent']
                    api_output_tokens_after = self.training_metrics['api_tokens_received']
                    reward_api_input_tokens = api_input_tokens_after - api_input_tokens_before
                    reward_api_output_tokens = api_output_tokens_after - api_output_tokens_before
                    reward_tokens_per_sec = (reward_api_input_tokens + reward_api_output_tokens) / reward_time if reward_time > 0 else 0

                    # Teacher call deltas for this batch
                    teacher_gen_calls_batch = int(self.cache_stats.get("teacher_gen_calls", 0)) - teacher_gen_calls_before
                    teacher_gen_hits_batch = int(self.cache_stats.get("teacher_gen_cache_hits", 0)) - teacher_gen_hits_before
                    teacher_score_calls_batch = int(self.cache_stats.get("teacher_score_calls", 0)) - teacher_score_calls_before
                    teacher_score_hits_batch = int(self.cache_stats.get("teacher_score_cache_hits", 0)) - teacher_score_hits_before
                    
                    # Track epoch-level metrics
                    epoch_reward_times.append(reward_time)
                    epoch_reward_tokens.append(reward_api_input_tokens)
                    epoch_reward_output_tokens.append(reward_api_output_tokens)
                    
                    # Log cache hit rate for debugging
                    if batch_idx % 5 == 0 and len(samples) > 0:
                        # Estimate cache hits: if no API tokens were used but we have rewards, likely all cached
                        if reward_api_tokens == 0 and len(rewards) > 0:
                            logger.debug(f"Batch {batch_idx}: All rewards from cache (no API calls)")
                        elif reward_api_tokens > 0:
                            logger.debug(f"Batch {batch_idx}: {reward_api_tokens} API tokens used for scoring")
                    
                    epoch_rewards.extend(rewards)

                    # Best-of-N per prompt (robust metric when you increase num_samples_per_prompt):
                    # mean reward can drop as you add more exploratory samples, even if the best sample improves.
                    try:
                        best_by_prompt: Dict[str, float] = {}
                        for s, r in zip(samples, rewards):
                            p = s.get("prompt")
                            if p is None:
                                continue
                            rr = float(r)
                            if p not in best_by_prompt or rr > best_by_prompt[p]:
                                best_by_prompt[p] = rr
                        if best_by_prompt:
                            epoch_best_reward_per_prompt.append(float(np.mean(list(best_by_prompt.values()))))
                    except Exception:
                        pass
                    
                    # Collect generated codes for diversity analysis
                    for sample in samples:
                        if 'code' in sample:
                            epoch_generated_codes.append(sample['code'])
                    
                    # Batch dataset collection to reduce memory overhead
                    dataset_batch.extend(dataset_entries)
                    if len(dataset_batch) >= dataset_batch_size * len(samples):
                        self.dataset_collection['training'].extend(dataset_batch)
                        dataset_batch = []  # Clear batch
                    
                    # Training step
                    # Only train if we have rewards and samples
                    if rewards and len(rewards) > 0 and samples and len(samples) > 0:
                        # Store rewards so `_create_training_batch_from_samples` can pick the best sample per prompt.
                        # (We keep it ephemeral to avoid threading complexity.)
                        self._latest_batch_rewards = rewards

                        # Reconstruct batch from generated samples (with full sequences: prompt + generated code)
                        # The original batch only has prompts, but we need the full sequences for training
                        train_batch = self._create_training_batch_from_samples(samples, batch['prompt'])
                        
                        train_start = time.time()
                        # Ensure rewards list is long enough (train_batch uses one sample per prompt)
                        rewards_for_training = rewards[:len(batch['prompt'])] if len(rewards) >= len(batch['prompt']) else rewards + [0.0] * (len(batch['prompt']) - len(rewards))
                        loss_dict = self.train_step(train_batch, rewards_for_training)
                        train_time = max(time.time() - train_start, 0.001)  # Ensure non-zero (min 1ms for display)
                        epoch_losses.append(loss_dict['loss'])
                        
                        # Calculate training tokens/sec (tokens processed during forward+backward pass)
                        # This is the number of tokens in the input sequence
                        train_num_tokens = train_batch['input_ids'].numel() if 'input_ids' in train_batch else 0
                        train_tokens_per_sec = train_num_tokens / train_time if train_time > 0 else 0
                        
                        # Track epoch-level metrics
                        epoch_train_times.append(train_time)
                        epoch_train_tokens.append(train_num_tokens)

                        # ---- Optimizer stepping / gradient accumulation (CRITICAL) ----
                        micro_step_in_epoch += 1
                        if micro_step_in_epoch % self.config.gradient_accumulation_steps == 0:
                            has_gradients = any(p.grad is not None for p in self.model.parameters() if p.requires_grad)
                            if not has_gradients:
                                logger.warning(f"⚠️  No gradients found at micro_step {micro_step_in_epoch}! Model may not be learning.")

                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1

                            # Stats/logging/checkpointing should happen on optimizer steps (not only at epoch flush).
                            if rewards and len(rewards) > 0:
                                self.stats['step'] = global_step
                                self.stats['total_reward'] += float(np.mean(rewards))
                                self.stats['num_samples'] += int(len(rewards))
                            if loss_dict and loss_dict.get('loss', 0.0) > 0:
                                self.stats['total_loss'] += float(loss_dict.get('loss', 0.0))
                            self.stats['avg_reward'] = self.stats['total_reward'] / max(1, self.stats.get('step', 1))
                            self.stats['avg_loss'] = self.stats['total_loss'] / max(1, self.stats.get('step', 1))

                            if global_step % self.config.logging_steps == 0:
                                if loss_dict and rewards:
                                    self._log_stats(global_step, loss_dict, rewards)

                            # Standard step-based checkpoint (but make the dir name unique)
                            if global_step % self.config.save_steps == 0:
                                ckpt_name = f"checkpoint-gs{global_step}-e{epoch+1}-b{batch_idx}"
                                self._save_checkpoint(
                                    global_step,
                                    checkpoint_name=ckpt_name,
                                    summary={
                                        "kind": "checkpoint",
                                        "reason": "save_steps",
                                        "epoch": int(epoch + 1),
                                        "batch_idx": int(batch_idx),
                                        "global_step": int(global_step),
                                        "avg_reward": float(np.mean(epoch_rewards)) if epoch_rewards else float(np.mean(rewards)) if rewards else 0.0,
                                        "avg_loss": float(np.mean(epoch_losses)) if epoch_losses else float(loss_dict.get("loss", 0.0)),
                                    },
                                )

                    # Optional: batch-based checkpointing for long epochs (captures batch context)
                    if int(getattr(self.config, "save_every_batches", 0) or 0) > 0:
                        n = int(self.config.save_every_batches)
                        if batch_idx > 0 and (batch_idx % n) == 0:
                            avg_so_far = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
                            ckpt_name = f"checkpoint-e{epoch+1}-b{batch_idx}-gs{global_step}"
                            self._save_checkpoint(
                                global_step,
                                checkpoint_name=ckpt_name,
                                summary={
                                    "kind": "checkpoint",
                                    "reason": "save_every_batches",
                                    "epoch": int(epoch + 1),
                                    "batch_idx": int(batch_idx),
                                    "global_step": int(global_step),
                                    "avg_reward": avg_so_far,
                                    "avg_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                                },
                            )

                            if global_step % 10 == 0:
                                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else self.config.learning_rate
                                logger.debug(
                                    f"Optimizer step {global_step}: LR={current_lr:.2e}, "
                                    f"grad_accum={self.config.gradient_accumulation_steps}, "
                                    f"micro_step_in_epoch={micro_step_in_epoch}"
                                )
                    else:
                        # No rewards or samples, skip training but set defaults
                        train_time = 0.001
                        train_num_tokens = 0
                        train_tokens_per_sec = 0
                        loss_dict = {'loss': 0.0, 'policy_loss': 0.0, 'kl_penalty': 0.0, 'avg_reward': 0.0}
                
                batch_time = time.time() - batch_start_time
                
                # Log timing info periodically with more detail
                if batch_idx % 5 == 0:
                    batch_size_actual = len(batch['prompt'])
                    # Use higher precision for small times to avoid showing 0.0s
                    gen_time_str = f"{gen_time:.3f}s" if gen_time < 1.0 else f"{gen_time:.1f}s"
                    reward_time_str = f"{reward_time:.3f}s" if reward_time < 1.0 else f"{reward_time:.1f}s"
                    train_time_str = f"{train_time:.3f}s" if train_time < 1.0 else f"{train_time:.1f}s"

                    # Generation: report raw vs kept, avg tokens/sample, and per-sample vs overall tok/s
                    raw_samples = before_n if 'before_n' in locals() else (len(samples_all) if 'samples_all' in locals() else 0)
                    kept_samples = after_n if 'after_n' in locals() else (len(samples) if 'samples' in locals() else 0)
                    avg_tok_per_sample_raw = (raw_num_tokens / raw_samples) if raw_samples > 0 else 0.0
                    avg_tok_per_sample_kept = (num_tokens / kept_samples) if kept_samples > 0 else 0.0

                    # Per-sample tok/s (average of per-sample tok/s) if available; otherwise fallback to overall tok/s
                    def _avg_sample_tps(sample_list):
                        vals = []
                        for s in (sample_list or []):
                            ot = s.get('output_tokens', 0) or 0
                            ts = s.get('gen_seconds', 0.0) or 0.0
                            if ot > 0 and ts > 0:
                                vals.append(ot / ts)
                        return float(np.mean(vals)) if vals else None

                    raw_sample_tps = _avg_sample_tps(samples_all if 'samples_all' in locals() else [])
                    kept_sample_tps = _avg_sample_tps(samples if 'samples' in locals() else [])
                    raw_sample_tps_str = f"{raw_sample_tps:.1f}" if raw_sample_tps is not None else "n/a"
                    kept_sample_tps_str = f"{kept_sample_tps:.1f}" if kept_sample_tps is not None else "n/a"
                    
                    # Add cache status to scoring line if applicable
                    scoring_line = f"  Scoring: {reward_time_str} ({reward_time/batch_time*100:.1f}%), {reward_api_tokens:,} API tokens, {reward_tokens_per_sec:.1f} tokens/sec"
                    # Split teacher call stats (generation vs scoring) for this batch if available
                    if 'teacher_gen_calls_batch' in locals():
                        scoring_line += (
                            f" | TeacherGenCalls {int(teacher_gen_calls_batch)} (hits {int(teacher_gen_hits_batch)})"
                            f" | TeacherScoreCalls {int(teacher_score_calls_batch)} (hits {int(teacher_score_hits_batch)})"
                        )
                    # Check if samples exists (it should always exist at this point)
                    samples_len = len(samples) if 'samples' in locals() and samples is not None else 0
                    if reward_api_tokens == 0 and samples_len > 0:
                        scoring_line += " (all cached)"
                    elif samples_len == 0:
                        scoring_line += " (no samples)"
                    
                    # Add error information
                    error_info = ""
                    if generation_error:
                        error_info += f"  ⚠️  Generation errors: 1 (this batch)\n"
                    # Calculate current epoch scoring errors for display
                    current_epoch_scoring_errors = self.error_stats['scoring_errors'] - epoch_start_scoring_errors
                    if current_epoch_scoring_errors > 0:
                        error_info += f"  ⚠️  Scoring errors (epoch so far): {current_epoch_scoring_errors}\n"
                    
                    logger.info(
                        f"Batch {batch_idx} (size={batch_size_actual}) timing breakdown:\n"
                        f"  Generation: {gen_time_str} ({gen_time/batch_time*100:.1f}%)\n"
                        f"    samples: raw={raw_samples} kept={kept_samples} | avg tok/sample: raw={avg_tok_per_sample_raw:.1f} kept={avg_tok_per_sample_kept:.1f}\n"
                        f"    tok/s: raw overall={raw_tokens_per_sec:.1f} avg-per-sample={raw_sample_tps_str} | kept overall={tokens_per_sec:.1f} avg-per-sample={kept_sample_tps_str}\n"
                        f"{scoring_line}\n"
                        f"  Training: {train_time_str} ({train_time/batch_time*100:.1f}%), {train_num_tokens:,} tokens, {train_tokens_per_sec:.1f} tokens/sec\n"
                        f"  Total: {batch_time:.1f}s"
                        + (f"\n{error_info.rstrip()}" if error_info else "")
                    )
                    
                    # Identify bottleneck
                    if gen_time > batch_time * 0.5:
                        logger.warning(f"⚠️  Generation is the bottleneck ({gen_time/batch_time*100:.1f}% of time)")
                        if self.mlx_model is None:
                            logger.warning("  → Enable MLX for 5-10x speedup (see above)")
                    elif reward_time > batch_time * 0.5:
                        logger.warning(f"⚠️  Scoring is the bottleneck ({reward_time/batch_time*100:.1f}% of time)")
                        logger.info("  → Consider reducing num_samples_per_prompt or increasing API parallelism")
                    elif train_time > batch_time * 0.5:
                        logger.warning(f"⚠️  Training step is the bottleneck ({train_time/batch_time*100:.1f}% of time)")
                        logger.info("  → Consider reducing batch_size or max_length")

                # --- Offline JSON summaries (every batch) ---
                try:
                    gen_backend = "mlx" if (self.mlx_model is not None and self.mlx_tokenizer is not None) else ("unsloth" if self._unsloth_enabled else "pytorch")
                    train_backend = "unsloth" if self._unsloth_enabled else "pytorch"
                    batch_size_actual = len(batch.get('prompt', [])) if isinstance(batch, dict) else 0
                    raw_samples = int(before_n) if 'before_n' in locals() else int(len(samples_all) if 'samples_all' in locals() else 0)
                    kept_samples = int(after_n) if 'after_n' in locals() else int(len(samples) if 'samples' in locals() else 0)
                    dup_filtered = max(0, raw_samples - kept_samples)
                    diversity_ratio = (kept_samples / raw_samples) if raw_samples > 0 else 0.0

                    # Avg-per-sample tok/s (mean over per-call tok/s if available)
                    def _mean_sample_tps(sample_list):
                        vals = []
                        for s in (sample_list or []):
                            ot = int(s.get('output_tokens', 0) or 0)
                            ts = float(s.get('gen_seconds', 0.0) or 0.0)
                            if ot > 0 and ts > 0:
                                vals.append(ot / ts)
                        return float(np.mean(vals)) if vals else 0.0

                    raw_sample_tps = _mean_sample_tps(samples_all if 'samples_all' in locals() else [])
                    kept_sample_tps = _mean_sample_tps(samples if 'samples' in locals() else [])

                    reward_api_input_tokens = int(reward_api_input_tokens) if 'reward_api_input_tokens' in locals() else 0
                    reward_api_output_tokens = int(reward_api_output_tokens) if 'reward_api_output_tokens' in locals() else 0
                    reward_api_total_tokens = reward_api_input_tokens + reward_api_output_tokens

                    rewards_mean = float(np.mean(rewards)) if rewards else 0.0
                    rewards_min = float(np.min(rewards)) if rewards else 0.0
                    rewards_max = float(np.max(rewards)) if rewards else 0.0
                    rewards_var = float(np.var(rewards)) if rewards and len(rewards) > 1 else 0.0

                    # Best-of-N per prompt (batch-level)
                    best_by_prompt: Dict[str, float] = {}
                    try:
                        for s, r in zip((samples or []), (rewards or [])):
                            p = s.get("prompt")
                            if not p:
                                continue
                            rr = float(r)
                            if p not in best_by_prompt or rr > best_by_prompt[p]:
                                best_by_prompt[p] = rr
                    except Exception:
                        best_by_prompt = {}
                    avg_best_per_prompt = float(np.mean(list(best_by_prompt.values()))) if best_by_prompt else None

                    # Reward gain tracking (vs baseline and vs previous batch)
                    baseline = float(self.baseline_reward) if self.baseline_reward is not None else 0.0
                    gain_from_baseline = rewards_mean - baseline if self.baseline_reward is not None else 0.0
                    prev = float(self._prev_batch_avg_reward) if self._prev_batch_avg_reward is not None else None
                    gain_vs_prev = (rewards_mean - prev) if prev is not None else 0.0

                    # EMA smoothing helps compare "improving with more samples" when batch reward is noisy.
                    ema_alpha = 0.2
                    if self._reward_ema is None:
                        self._reward_ema = rewards_mean
                    else:
                        self._reward_ema = (1.0 - ema_alpha) * float(self._reward_ema) + ema_alpha * rewards_mean
                    ema = float(self._reward_ema)
                    ema_gain_from_baseline = (ema - baseline) if self.baseline_reward is not None else 0.0

                    sysm = {}
                    try:
                        sysm = self._get_system_metrics() or {}
                    except Exception:
                        sysm = {}

                    self._log_batch_json({
                        "kind": "batch",
                        "run_id": os.environ.get("RUN_ID") or None,
                        "model": self.config.base_model,
                        "device": str(self.device),
                        "generation_backend": gen_backend,
                        "training_backend": train_backend,
                        "epoch": int(epoch + 1),
                        "batch_idx": int(batch_idx),
                        "global_step": int(global_step),
                        "micro_step_in_epoch": int(micro_step_in_epoch),
                        "batch_size": int(batch_size_actual),
                        "timing_s": {
                            "generation": float(gen_time),
                            "scoring": float(reward_time),
                            "training": float(train_time),
                            "total": float(batch_time),
                        },
                        "tokens": {
                            "gen_raw": int(raw_num_tokens) if 'raw_num_tokens' in locals() else 0,
                            "gen_kept": int(num_tokens) if 'num_tokens' in locals() else 0,
                            "train_input": int(train_num_tokens) if 'train_num_tokens' in locals() else 0,
                            "scoring_api_input": int(reward_api_input_tokens),
                            "scoring_api_output": int(reward_api_output_tokens),
                            "scoring_api_total": int(reward_api_total_tokens),
                        },
                        "throughput_tok_s": {
                            "gen_raw_overall": float(raw_tokens_per_sec) if 'raw_tokens_per_sec' in locals() else 0.0,
                            "gen_kept_overall": float(tokens_per_sec) if 'tokens_per_sec' in locals() else 0.0,
                            "gen_raw_avg_per_sample": float(raw_sample_tps),
                            "gen_kept_avg_per_sample": float(kept_sample_tps),
                            "scoring_api_total": float(reward_tokens_per_sec) if 'reward_tokens_per_sec' in locals() else 0.0,
                            "training": float(train_tokens_per_sec) if 'train_tokens_per_sec' in locals() else 0.0,
                        },
                        "samples": {
                            "raw": int(raw_samples),
                            "kept": int(kept_samples),
                            "dup_filtered": int(dup_filtered),
                            "diversity_ratio": float(diversity_ratio),
                            "avg_tok_per_sample_raw": float((raw_num_tokens / raw_samples) if raw_samples > 0 else 0.0) if 'raw_num_tokens' in locals() else 0.0,
                            "avg_tok_per_sample_kept": float((num_tokens / kept_samples) if kept_samples > 0 else 0.0) if 'num_tokens' in locals() else 0.0,
                        },
                        "rewards": {
                            "mean": rewards_mean,
                            "min": rewards_min,
                            "max": rewards_max,
                            "var": rewards_var,
                            "count": int(len(rewards)) if rewards else 0,
                        },
                        "rewards_best_of_n": {
                            "n": int(self.config.num_samples_per_prompt),
                            "avg_best_per_prompt": avg_best_per_prompt,
                        },
                        "reward_gain": {
                            "baseline_reward": float(baseline) if self.baseline_reward is not None else None,
                            "gain_from_baseline": float(gain_from_baseline) if self.baseline_reward is not None else None,
                            "prev_batch_reward": float(prev) if prev is not None else None,
                            "gain_vs_prev_batch": float(gain_vs_prev) if prev is not None else None,
                            "ema_reward": float(ema),
                            "ema_gain_from_baseline": float(ema_gain_from_baseline) if self.baseline_reward is not None else None,
                            "ema_alpha": float(ema_alpha),
                        },
                        "loss": {
                            "loss": float(loss_dict.get("loss", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                            "policy_loss": float(loss_dict.get("policy_loss", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                            "kl_penalty": float(loss_dict.get("kl_penalty", 0.0)) if isinstance(loss_dict, dict) else 0.0,
                        },
                        "cache": {
                            "api_calls_total": int(self.cache_stats.get("api_calls", 0)),
                            "cache_hits_total": int(self.cache_stats.get("cache_hits", 0)),
                            "cache_misses_total": int(self.cache_stats.get("cache_misses", 0)),
                        },
                        "errors": {
                            "generation_error_batch": bool(generation_error),
                            "generation_errors_total": int(self.error_stats.get("generation_errors", 0)),
                            "scoring_errors_total": int(self.error_stats.get("scoring_errors", 0)),
                            "teacher_generate_errors_total": int(self.error_stats.get("teacher_generate_errors", 0)),
                            "teacher_scoring_errors_total": int(self.error_stats.get("teacher_scoring_errors", 0)),
                        },
                        "system": sysm,
                    })

                    # Update previous batch reward after logging
                    self._prev_batch_avg_reward = rewards_mean
                except Exception:
                    pass

                # --- TensorBoard per-batch time series (continuous) ---
                try:
                    if self.writer:
                        interval = int(getattr(self.config, "tensorboard_batch_interval", 1) or 1)
                        if interval < 1:
                            interval = 1
                        # Increment batch step (monotonic) and log at the chosen interval.
                        self._batch_step += 1
                        if (self._batch_step % interval) == 0:
                            bs = int(self._batch_step)
                            # Safe defaults (some vars only exist if rewards/scoring ran)
                            _teacher_in = float(reward_api_input_tokens) if 'reward_api_input_tokens' in locals() else 0.0
                            _teacher_out = float(reward_api_output_tokens) if 'reward_api_output_tokens' in locals() else 0.0
                            _gen_raw_tok = float(raw_num_tokens) if 'raw_num_tokens' in locals() else 0.0
                            _gen_kept_tok = float(num_tokens) if 'num_tokens' in locals() else 0.0

                            # Core reward signals
                            self.writer.add_scalar("Batch/Reward_Mean", rewards_mean, bs)
                            if avg_best_per_prompt is not None:
                                self.writer.add_scalar("Batch/Reward_BestOfN_PerPrompt", float(avg_best_per_prompt), bs)
                            if self.baseline_reward is not None:
                                self.writer.add_scalar("Batch/Reward_GainFromBaseline", float(gain_from_baseline), bs)
                                self.writer.add_scalar("Batch/Reward_EMA_GainFromBaseline", float(ema_gain_from_baseline), bs)
                            self.writer.add_scalar("Batch/Reward_EMA", float(ema), bs)

                            # Generation shape/efficiency
                            self.writer.add_scalar("Batch/Gen_Samples_Raw", float(raw_samples), bs)
                            self.writer.add_scalar("Batch/Gen_Samples_Kept", float(kept_samples), bs)
                            self.writer.add_scalar("Batch/Gen_DiversityRatio", float(diversity_ratio), bs)
                            self.writer.add_scalar("Batch/Gen_TokPerSample_Raw", float((_gen_raw_tok / raw_samples) if raw_samples > 0 else 0.0), bs)
                            self.writer.add_scalar("Batch/Gen_TokPerSample_Kept", float((_gen_kept_tok / kept_samples) if kept_samples > 0 else 0.0), bs)
                            self.writer.add_scalar("Batch/Gen_TokPerSec_RawOverall", float(raw_tokens_per_sec) if 'raw_tokens_per_sec' in locals() else 0.0, bs)
                            self.writer.add_scalar("Batch/Gen_TokPerSec_KeptOverall", float(tokens_per_sec) if 'tokens_per_sec' in locals() else 0.0, bs)
                            self.writer.add_scalar("Batch/Gen_TokPerSec_RawAvgPerSample", float(raw_sample_tps), bs)
                            self.writer.add_scalar("Batch/Gen_TokPerSec_KeptAvgPerSample", float(kept_sample_tps), bs)

                            # Phase times and throughput
                            self.writer.add_scalar("Batch/Time_Generation_s", float(gen_time), bs)
                            self.writer.add_scalar("Batch/Time_Scoring_s", float(reward_time), bs)
                            self.writer.add_scalar("Batch/Time_Training_s", float(train_time), bs)
                            self.writer.add_scalar("Batch/Time_Total_s", float(batch_time), bs)
                            self.writer.add_scalar("Batch/Training_TokPerSec", float(train_tokens_per_sec) if 'train_tokens_per_sec' in locals() else 0.0, bs)
                            self.writer.add_scalar("Batch/Scoring_TokPerSec_Total", float(reward_tokens_per_sec) if 'reward_tokens_per_sec' in locals() else 0.0, bs)

                            # Teacher calls (per batch) + tokens
                            if 'teacher_gen_calls_batch' in locals():
                                self.writer.add_scalar("Batch/TeacherGenCalls", float(teacher_gen_calls_batch), bs)
                                self.writer.add_scalar("Batch/TeacherGenCacheHits", float(teacher_gen_hits_batch), bs)
                                self.writer.add_scalar("Batch/TeacherScoreCalls", float(teacher_score_calls_batch), bs)
                                self.writer.add_scalar("Batch/TeacherScoreCacheHits", float(teacher_score_hits_batch), bs)
                            self.writer.add_scalar("Batch/TeacherTokens_Input", _teacher_in, bs)
                            self.writer.add_scalar("Batch/TeacherTokens_Output", _teacher_out, bs)

                            # Training params trend (LR) if scheduler exists
                            try:
                                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else None
                                if current_lr is not None:
                                    self.writer.add_scalar("Batch/LR", float(current_lr), bs)
                            except Exception:
                                pass

                            # Loss scalars (batch)
                            if isinstance(loss_dict, dict):
                                self.writer.add_scalar("Batch/Loss", float(loss_dict.get("loss", 0.0)), bs)
                                self.writer.add_scalar("Batch/PolicyLoss", float(loss_dict.get("policy_loss", 0.0)), bs)
                                self.writer.add_scalar("Batch/KLPenalty", float(loss_dict.get("kl_penalty", 0.0)), bs)
                except Exception:
                    # Never fail training due to metrics logging
                    pass
                
                # Aggressive memory cleanup after training step to prevent OOM
                # Do this AFTER all logging is complete to avoid referencing deleted variables
                if torch.backends.mps.is_available():
                    # Delete intermediate tensors explicitly
                    if 'train_batch' in locals() and train_batch is not None:
                        del train_batch
                    if 'samples' in locals() and samples is not None:
                        del samples
                    if 'rewards' in locals() and rewards is not None:
                        del rewards
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Clear MPS cache aggressively
                    torch.mps.empty_cache()
                    
                    # Synchronize to ensure cleanup is complete
                    torch.mps.synchronize()
                    
                    logger.debug(f"Memory cleanup complete after training step {batch_idx}")
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Flush remaining dataset entries at end of epoch
            if dataset_batch:
                self.dataset_collection['training'].extend(dataset_batch)
                dataset_batch = []

            # If we have leftover accumulated gradients at epoch end, flush them so we still learn on small datasets.
            if micro_step_in_epoch > 0 and (micro_step_in_epoch % self.config.gradient_accumulation_steps) != 0:
                has_gradients = any(p.grad is not None for p in self.model.parameters() if p.requires_grad)
                if has_gradients:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    logger.debug(
                        f"Flushed leftover grads at epoch end: micro_step_in_epoch={micro_step_in_epoch}, "
                        f"grad_accum={self.config.gradient_accumulation_steps}"
                    )

                    # Stats/logging/checkpointing on flush step
                    if rewards and len(rewards) > 0:
                        self.stats['step'] = global_step
                        self.stats['total_reward'] += float(np.mean(rewards))
                        self.stats['num_samples'] += int(len(rewards))
                    if loss_dict and loss_dict.get('loss', 0.0) > 0:
                        self.stats['total_loss'] += float(loss_dict.get('loss', 0.0))
                    self.stats['avg_reward'] = self.stats['total_reward'] / max(1, self.stats.get('step', 1))
                    self.stats['avg_loss'] = self.stats['total_loss'] / max(1, self.stats.get('step', 1))

                    if global_step % self.config.logging_steps == 0:
                        if loss_dict and rewards:
                            self._log_stats(global_step, loss_dict, rewards)

                    if global_step % self.config.save_steps == 0:
                        ckpt_name = f"checkpoint-gs{global_step}-e{epoch+1}-flush"
                        self._save_checkpoint(
                            global_step,
                            checkpoint_name=ckpt_name,
                            summary={
                                "kind": "checkpoint",
                                "reason": "save_steps_flush",
                                "epoch": int(epoch + 1),
                                "batch_idx": int(batch_idx),
                                "global_step": int(global_step),
                                "avg_reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
                                "avg_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                            },
                        )
                
                # (Moved) stats/logging/checkpointing are handled on optimizer steps above.
            
            # Epoch summary
            avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
            avg_epoch_best_reward_per_prompt = float(np.mean(epoch_best_reward_per_prompt)) if epoch_best_reward_per_prompt else 0.0
            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            
            # Track reward variance (lower is better - more consistent)
            reward_variance = np.var(epoch_rewards) if len(epoch_rewards) > 1 else 0.0
            
            # Track metrics for trend analysis
            self.training_metrics['reward_by_epoch'].append(avg_epoch_reward)
            self.training_metrics['loss_by_epoch'].append(avg_epoch_loss)
            self.training_metrics['reward_variance_by_epoch'].append(reward_variance)
            
            # Calculate reward and loss trends (change from previous epoch)
            reward_trend = 0.0
            loss_trend = 0.0
            if len(self.training_metrics['reward_by_epoch']) > 1:
                reward_trend = avg_epoch_reward - self.training_metrics['reward_by_epoch'][-2]
                loss_trend = self.training_metrics['loss_by_epoch'][-2] - avg_epoch_loss  # Loss should decrease
            
            # Track API tokens for this epoch (input and output separately)
            epoch_input_tokens = self.training_metrics['api_tokens_sent'] - epoch_start_api_tokens
            if epoch == 0:
                epoch_output_tokens = self.training_metrics['api_tokens_received']
            else:
                # Calculate output tokens for this epoch from cumulative
                prev_output_tokens = sum(self.training_metrics['api_output_tokens_by_epoch']) if self.training_metrics['api_output_tokens_by_epoch'] else 0
                epoch_output_tokens = self.training_metrics['api_tokens_received'] - prev_output_tokens
            
            self.training_metrics['api_tokens_by_epoch'].append(epoch_input_tokens)
            self.training_metrics['api_output_tokens_by_epoch'].append(epoch_output_tokens)
            
            # Track teacher call/caching statistics for this epoch
            epoch_teacher_gen_calls = int(self.cache_stats.get('teacher_gen_calls', 0) - epoch_start_teacher_gen_calls)
            epoch_teacher_gen_cache_hits = int(self.cache_stats.get('teacher_gen_cache_hits', 0) - epoch_start_teacher_gen_cache_hits)
            epoch_teacher_score_calls = int(self.cache_stats.get('teacher_score_calls', 0) - epoch_start_teacher_score_calls)
            epoch_teacher_score_cache_hits = int(self.cache_stats.get('teacher_score_cache_hits', 0) - epoch_start_teacher_score_cache_hits)

            # Back-compat: these were historically "teacher gen" stats
            self.training_metrics['api_calls_by_epoch'].append(epoch_teacher_gen_calls)
            self.training_metrics['cache_hits_by_epoch'].append(epoch_teacher_gen_cache_hits)
            
            # Track error statistics for this epoch
            epoch_gen_errors = self.error_stats['generation_errors'] - epoch_start_gen_errors
            epoch_scoring_errors = self.error_stats['scoring_errors'] - epoch_start_scoring_errors
            epoch_teacher_generate_errors = int(self.error_stats.get('teacher_generate_errors', 0) - epoch_start_teacher_generate_errors)
            epoch_teacher_scoring_errors = int(self.error_stats.get('teacher_scoring_errors', 0) - epoch_start_teacher_scoring_errors)
            self.error_stats['generation_errors_by_epoch'].append(epoch_gen_errors)
            self.error_stats['scoring_errors_by_epoch'].append(epoch_scoring_errors)
            self.error_stats['teacher_generate_errors_by_epoch'].append(epoch_teacher_generate_errors)
            self.error_stats['teacher_scoring_errors_by_epoch'].append(epoch_teacher_scoring_errors)
            
            # Calculate code diversity metrics
            code_diversity = self._calculate_code_diversity(epoch_generated_codes)
            self.training_metrics['code_diversity_by_epoch'].append(code_diversity)
            
            # Calculate scoring breakdown for this epoch from dataset entries
            # Extract scoring breakdown from dataset entries collected this epoch
            epoch_scoring_breakdown = {
                'correctness': [],
                'code_quality': [],
                'efficiency': [],
                'documentation': []
            }
            
            # Get dataset entries from this epoch (from dataset_batch or dataset_collection)
            epoch_dataset_entries = []
            if dataset_batch:
                epoch_dataset_entries.extend(dataset_batch)
            # Also check if we can get from dataset_collection
            if len(self.dataset_collection.get('training', [])) > 0:
                # Get entries from this epoch (approximate by checking recent entries)
                recent_entries = self.dataset_collection['training'][-len(epoch_rewards):] if len(epoch_rewards) > 0 else []
                epoch_dataset_entries.extend(recent_entries)
            
            # Extract scoring breakdown (weights are fixed, but we track average scores)
            # For now, we'll use the fixed weights as placeholders and track actual scores
            # The actual breakdown would require parsing teacher responses, which is complex
            # So we'll use the fixed weights and note that actual scores vary
            epoch_scoring_breakdown_avg = {
                'correctness': 0.3,  # Weight
                'code_quality': 0.3,  # Weight
                'efficiency': 0.2,  # Weight
                'documentation': 0.2  # Weight
            }
            self.training_metrics['scoring_breakdown_by_epoch'].append(epoch_scoring_breakdown_avg)
            
            # Cache hit rates (separate: teacher reference generation vs scoring)
            gen_ops = epoch_teacher_gen_calls + epoch_teacher_gen_cache_hits
            score_ops = epoch_teacher_score_calls + epoch_teacher_score_cache_hits
            teacher_gen_cache_hit_rate = (epoch_teacher_gen_cache_hits / gen_ops * 100) if gen_ops > 0 else 0.0
            teacher_score_cache_hit_rate = (epoch_teacher_score_cache_hits / score_ops * 100) if score_ops > 0 else 0.0

            # Back-compat aggregate cache hit rate (used by older logs/TensorBoard).
            total_ops = gen_ops + score_ops
            cache_hit_rate = ((epoch_teacher_gen_cache_hits + epoch_teacher_score_cache_hits) / total_ops * 100) if total_ops > 0 else 0.0
            
            # Calculate average performance metrics
            # Average tokens/sec = total tokens / total time
            total_gen_time = sum(epoch_gen_times) if epoch_gen_times else 0.0
            total_reward_time = sum(epoch_reward_times) if epoch_reward_times else 0.0
            total_train_time = sum(epoch_train_times) if epoch_train_times else 0.0
            
            total_gen_tokens = sum(epoch_gen_tokens) if epoch_gen_tokens else 0
            total_gen_tokens_raw = sum(epoch_gen_tokens_raw) if epoch_gen_tokens_raw else 0
            total_gen_samples_raw = sum(epoch_gen_samples_raw) if epoch_gen_samples_raw else 0
            total_gen_samples_kept = sum(epoch_gen_samples_kept) if epoch_gen_samples_kept else 0
            total_train_tokens = sum(epoch_train_tokens) if epoch_train_tokens else 0
            total_reward_input_tokens = sum(epoch_reward_tokens) if epoch_reward_tokens else 0
            total_reward_output_tokens = sum(epoch_reward_output_tokens) if epoch_reward_output_tokens else 0
            
            # Calculate average token size generated (output tokens per sample)
            num_samples = len(epoch_rewards) if epoch_rewards else 1
            avg_tokens_per_sample = total_gen_tokens / num_samples if num_samples > 0 else 0.0
            avg_tokens_per_sample_raw = total_gen_tokens_raw / num_samples if num_samples > 0 else 0.0

            # Average tokens/sample measured over actual generated samples (raw/kept)
            avg_tok_per_gen_sample_raw = (total_gen_tokens_raw / total_gen_samples_raw) if total_gen_samples_raw > 0 else 0.0
            avg_tok_per_gen_sample_kept = (total_gen_tokens / total_gen_samples_kept) if total_gen_samples_kept > 0 else 0.0

            # Avg-per-sample tok/s across the epoch (raw vs kept)
            avg_gen_sample_tps_raw = float(np.mean(epoch_gen_sample_tps_raw)) if epoch_gen_sample_tps_raw else 0.0
            avg_gen_sample_tps_kept = float(np.mean(epoch_gen_sample_tps_kept)) if epoch_gen_sample_tps_kept else 0.0
            
            # Generation throughput:
            # - raw: all sampled tokens / gen_time  (true MLX throughput)
            # - kept: post-dedup tokens / gen_time  (effective training yield)
            avg_gen_tokens_per_sec_raw = total_gen_tokens_raw / total_gen_time if total_gen_time > 0 else 0.0
            avg_gen_tokens_per_sec = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0.0
            # For scoring, calculate tokens/sec using total tokens (input + output)
            total_reward_tokens = total_reward_input_tokens + total_reward_output_tokens
            avg_reward_tokens_per_sec = total_reward_tokens / total_reward_time if total_reward_time > 0 else 0.0
            avg_train_tokens_per_sec = total_train_tokens / total_train_time if total_train_time > 0 else 0.0
            
            # Calculate average latencies
            avg_gen_latency = np.mean(epoch_gen_times) if epoch_gen_times else 0.0
            avg_reward_latency = np.mean(epoch_reward_times) if epoch_reward_times else 0.0
            avg_train_latency = np.mean(epoch_train_times) if epoch_train_times else 0.0
            
            # Calculate total epoch time
            epoch_total_time = time.time() - epoch_start_time
            epoch_total_time_minutes = epoch_total_time / 60.0
            epoch_total_time_str = f"{epoch_total_time_minutes:.1f} min" if epoch_total_time_minutes >= 1.0 else f"{epoch_total_time:.1f} sec"
            
            # Calculate trainable parameter count
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_percentage = (trainable_params / total_params * 100) if total_params > 0 else 0.0
            
            # Format parameter counts
            if trainable_params >= 1e9:
                trainable_str = f"{trainable_params / 1e9:.2f}B"
            elif trainable_params >= 1e6:
                trainable_str = f"{trainable_params / 1e6:.2f}M"
            elif trainable_params >= 1e3:
                trainable_str = f"{trainable_params / 1e3:.2f}K"
            else:
                trainable_str = f"{trainable_params:,}"
            
            logger.info(
                f"Epoch {epoch + 1} Summary:\n"
                f"  Total Time: {epoch_total_time_str} ({epoch_total_time:.1f}s)\n"
                f"  Average Reward: {avg_epoch_reward:.4f} (mean over all sampled completions)\n"
                f"  Best-of-N Reward: {avg_epoch_best_reward_per_prompt:.4f} (avg of per-prompt max; N={self.config.num_samples_per_prompt})\n"
                f"  Average Loss: {avg_epoch_loss:.4f}\n"
                f"  Total Samples: {len(epoch_rewards)}\n"
                f"  Trainable Parameters: {trainable_str} ({trainable_percentage:.2f}% of {total_params:,} total)\n"
                f"  TeacherGenCalls: {epoch_teacher_gen_calls:,} calls, {epoch_teacher_gen_cache_hits:,} cache hits ({teacher_gen_cache_hit_rate:.1f}% hit rate)\n"
                f"  TeacherScoreCalls: {epoch_teacher_score_calls:,} calls, {epoch_teacher_score_cache_hits:,} cache hits ({teacher_score_cache_hit_rate:.1f}% hit rate)\n"
                f"  TeacherTokens: {epoch_input_tokens:,} input tokens, {epoch_output_tokens:,} output tokens\n"
                f"  Code Diversity: {code_diversity['unique_ratio']:.1%} unique ({code_diversity['unique_count']}/{code_diversity['total_count']}), avg similarity: {code_diversity['avg_similarity']:.3f}\n"
                f"  Errors: StudentGeneration: {epoch_gen_errors}, TeacherGenerate: {epoch_teacher_generate_errors}, TeacherScoring: {epoch_teacher_scoring_errors} (total teacher errors: {epoch_scoring_errors})\n"
                f"  Performance:\n"
                f"    Generation: raw {avg_gen_tokens_per_sec_raw:.1f} tok/s ({total_gen_tokens_raw:,} tokens) | "
                f"kept {avg_gen_tokens_per_sec:.1f} tok/s ({total_gen_tokens:,} tokens)\n"
                f"      samples: raw={total_gen_samples_raw} kept={total_gen_samples_kept} | "
                f"avg tok/sample: raw={avg_tok_per_gen_sample_raw:.1f} kept={avg_tok_per_gen_sample_kept:.1f}\n"
                f"      avg-per-sample tok/s: raw={avg_gen_sample_tps_raw:.1f} kept={avg_gen_sample_tps_kept:.1f} | "
                f"(avg batch latency: {avg_gen_latency:.3f}s)\n"
                f"    Scoring: {avg_reward_tokens_per_sec:.1f} tokens/sec (avg latency: {avg_reward_latency:.3f}s, input: {total_reward_input_tokens:,}, output: {total_reward_output_tokens:,})\n"
                f"    Training: {avg_train_tokens_per_sec:.1f} tokens/sec (avg latency: {avg_train_latency:.3f}s, total input sequence tokens: {total_train_tokens:,})"
            )

            # Save checkpoint at end of each epoch (independent of global_step)
            if int(getattr(self.config, "save_every_epochs", 0) or 0) > 0:
                n = int(self.config.save_every_epochs)
                if ((epoch + 1) % n) == 0:
                    ckpt_name = f"checkpoint-e{epoch+1}-end-gs{global_step}"
                    self._save_checkpoint(
                        global_step,
                        checkpoint_name=ckpt_name,
                        summary={
                            "kind": "checkpoint",
                            "reason": "epoch_end",
                            "epoch": int(epoch + 1),
                            "batch_idx": int(batch_idx),
                            "global_step": int(global_step),
                            "avg_reward": float(avg_epoch_reward),
                            "avg_loss": float(avg_epoch_loss),
                            "reward_variance": float(reward_variance),
                            "api_input_tokens": int(epoch_input_tokens),
                            "api_output_tokens": int(epoch_output_tokens),
                            "cache_hit_rate": float(cache_hit_rate),
                            "diversity_unique_ratio": float(code_diversity.get("unique_ratio", 0.0)),
                            "generation_tok_s_raw": float(avg_gen_tokens_per_sec_raw),
                            "generation_tok_s_kept": float(avg_gen_tokens_per_sec),
                            "training_tok_s": float(avg_train_tokens_per_sec),
                        },
                    )

            # --- Offline JSON summaries (every epoch) ---
            try:
                gen_backend = "mlx" if (self.mlx_model is not None and self.mlx_tokenizer is not None) else ("unsloth" if self._unsloth_enabled else "pytorch")
                train_backend = "unsloth" if self._unsloth_enabled else "pytorch"
                self._log_epoch_json({
                    "kind": "epoch",
                    "run_id": os.environ.get("RUN_ID") or None,
                    "model": self.config.base_model,
                    "device": str(self.device),
                    "generation_backend": gen_backend,
                    "training_backend": train_backend,
                    "epoch": int(epoch + 1),
                    "epoch_time_s": float(epoch_total_time),
                    "avg_reward": float(avg_epoch_reward),
                    "avg_loss": float(avg_epoch_loss),
                    "reward_variance": float(reward_variance),
                    "reward_trend": float(reward_trend),
                    "loss_trend": float(loss_trend),
                    "num_samples": int(len(epoch_rewards)),
                    "trainable_params": int(trainable_params),
                    "total_params": int(total_params),
                    "trainable_percentage": float(trainable_percentage),
                    "api": {
                        "input_tokens": int(epoch_input_tokens),
                        "output_tokens": int(epoch_output_tokens),
                        "teacher_gen_calls": int(epoch_teacher_gen_calls),
                        "teacher_gen_cache_hits": int(epoch_teacher_gen_cache_hits),
                        "teacher_gen_cache_hit_rate": float(teacher_gen_cache_hit_rate),
                        "teacher_score_calls": int(epoch_teacher_score_calls),
                        "teacher_score_cache_hits": int(epoch_teacher_score_cache_hits),
                        "teacher_score_cache_hit_rate": float(teacher_score_cache_hit_rate),
                        # Back-compat aggregate:
                        "cache_hit_rate": float(cache_hit_rate),
                    },
                    "errors": {
                        "generation": int(epoch_gen_errors),
                        "teacher_generate": int(epoch_teacher_generate_errors),
                        "teacher_scoring": int(epoch_teacher_scoring_errors),
                        "total_teacher": int(epoch_scoring_errors),
                    },
                    "diversity": {
                        "unique_ratio": float(code_diversity.get("unique_ratio", 0.0)),
                        "unique_count": int(code_diversity.get("unique_count", 0)),
                        "total_count": int(code_diversity.get("total_count", 0)),
                        "avg_similarity": float(code_diversity.get("avg_similarity", 0.0)),
                    },
                    "performance": {
                        "generation": {
                            "tok_s_raw": float(avg_gen_tokens_per_sec_raw),
                            "tok_s_kept": float(avg_gen_tokens_per_sec),
                            "tokens_raw": int(total_gen_tokens_raw),
                            "tokens_kept": int(total_gen_tokens),
                            "samples_raw": int(total_gen_samples_raw),
                            "samples_kept": int(total_gen_samples_kept),
                            "avg_tok_per_sample_raw": float(avg_tok_per_gen_sample_raw),
                            "avg_tok_per_sample_kept": float(avg_tok_per_gen_sample_kept),
                            "avg_per_sample_tok_s_raw": float(avg_gen_sample_tps_raw),
                            "avg_per_sample_tok_s_kept": float(avg_gen_sample_tps_kept),
                            "avg_latency_s": float(avg_gen_latency),
                        },
                        "scoring": {
                            "tok_s_total": float(avg_reward_tokens_per_sec),
                            "avg_latency_s": float(avg_reward_latency),
                            "input_tokens": int(total_reward_input_tokens),
                            "output_tokens": int(total_reward_output_tokens),
                        },
                        "training": {
                            "tok_s": float(avg_train_tokens_per_sec),
                            "avg_latency_s": float(avg_train_latency),
                            "input_tokens": int(total_train_tokens),
                        },
                    },
                })
            except Exception:
                pass

            # After reporting this epoch, roll epoch hashes into global hashes so the next epoch
            # treats repeats as “seen before” and triggers novelty retries during generation.
            if hasattr(self, "_epoch_code_hashes") and hasattr(self, "_global_code_hashes"):
                self._global_code_hashes.update(self._epoch_code_hashes)
            
            if self.writer:
                # Simplified epoch charts: log only what appears in the epoch summary.
                ep = int(epoch + 1)
                self.writer.add_scalar("Epoch/Time_Total_s", float(epoch_total_time), ep)

                # Rewards & loss (summary)
                self.writer.add_scalar("Epoch/Reward_Mean", float(avg_epoch_reward), ep)
                self.writer.add_scalar("Epoch/Reward_BestOfN_PerPrompt", float(avg_epoch_best_reward_per_prompt), ep)
                self.writer.add_scalar("Epoch/Loss_Mean", float(avg_epoch_loss), ep)
                self.writer.add_scalar("Epoch/RewardVariance", float(reward_variance), ep)
                self.writer.add_scalar("Epoch/TotalSamples", float(len(epoch_rewards)), ep)

                # Teacher calls/tokens (summary)
                self.writer.add_scalar("Epoch/TeacherGenCalls", float(epoch_teacher_gen_calls), ep)
                self.writer.add_scalar("Epoch/TeacherGenCacheHits", float(epoch_teacher_gen_cache_hits), ep)
                self.writer.add_scalar("Epoch/TeacherGenCacheHitRate", float(teacher_gen_cache_hit_rate), ep)
                self.writer.add_scalar("Epoch/TeacherScoreCalls", float(epoch_teacher_score_calls), ep)
                self.writer.add_scalar("Epoch/TeacherScoreCacheHits", float(epoch_teacher_score_cache_hits), ep)
                self.writer.add_scalar("Epoch/TeacherScoreCacheHitRate", float(teacher_score_cache_hit_rate), ep)
                self.writer.add_scalar("Epoch/TeacherTokens_Input", float(epoch_input_tokens), ep)
                self.writer.add_scalar("Epoch/TeacherTokens_Output", float(epoch_output_tokens), ep)

                # Diversity + errors (summary)
                self.writer.add_scalar("Epoch/CodeDiversity_UniqueRatio", float(code_diversity.get("unique_ratio", 0.0)), ep)
                self.writer.add_scalar("Epoch/CodeDiversity_AvgSimilarity", float(code_diversity.get("avg_similarity", 0.0)), ep)
                self.writer.add_scalar("Epoch/Errors_StudentGeneration", float(epoch_gen_errors), ep)
                self.writer.add_scalar("Epoch/Errors_TeacherGenerate", float(epoch_teacher_generate_errors), ep)
                self.writer.add_scalar("Epoch/Errors_TeacherScoring", float(epoch_teacher_scoring_errors), ep)

                # Performance (summary)
                self.writer.add_scalar("Epoch/Gen_TokPerSec_Raw", float(avg_gen_tokens_per_sec_raw), ep)
                self.writer.add_scalar("Epoch/Gen_TokPerSec_Kept", float(avg_gen_tokens_per_sec), ep)
                self.writer.add_scalar("Epoch/Gen_Tokens_Raw", float(total_gen_tokens_raw), ep)
                self.writer.add_scalar("Epoch/Gen_Tokens_Kept", float(total_gen_tokens), ep)
                self.writer.add_scalar("Epoch/Gen_Samples_Raw", float(total_gen_samples_raw), ep)
                self.writer.add_scalar("Epoch/Gen_Samples_Kept", float(total_gen_samples_kept), ep)
                self.writer.add_scalar("Epoch/Gen_TokPerSample_Raw", float(avg_tok_per_gen_sample_raw), ep)
                self.writer.add_scalar("Epoch/Gen_TokPerSample_Kept", float(avg_tok_per_gen_sample_kept), ep)
                self.writer.add_scalar("Epoch/Gen_TokPerSec_RawAvgPerSample", float(avg_gen_sample_tps_raw), ep)
                self.writer.add_scalar("Epoch/Gen_TokPerSec_KeptAvgPerSample", float(avg_gen_sample_tps_kept), ep)
                self.writer.add_scalar("Epoch/Gen_AvgBatchLatency_s", float(avg_gen_latency), ep)
                self.writer.add_scalar("Epoch/Scoring_TokPerSec_Total", float(avg_reward_tokens_per_sec), ep)
                self.writer.add_scalar("Epoch/Scoring_AvgLatency_s", float(avg_reward_latency), ep)
                self.writer.add_scalar("Epoch/Training_TokPerSec", float(avg_train_tokens_per_sec), ep)
                self.writer.add_scalar("Epoch/Training_AvgLatency_s", float(avg_train_latency), ep)
                self.writer.add_scalar("Epoch/Training_InputTokens", float(total_train_tokens), ep)
        
        # Record training end time
        self.training_metrics['training_end_time'] = time.time()
        
        # Final save (unique name to avoid overwriting checkpoint-0)
        self._save_checkpoint(
            global_step,
            final=True,
            checkpoint_name=f"checkpoint-final-gs{global_step}",
            summary={
                "kind": "checkpoint",
                "reason": "final",
                "epoch": int(self.stats.get("epoch", 0) or 0),
                "global_step": int(global_step),
                "avg_reward": float(self.training_metrics["reward_by_epoch"][-1]) if self.training_metrics.get("reward_by_epoch") else float(self.stats.get("avg_reward", 0.0)),
                "avg_loss": float(self.training_metrics["loss_by_epoch"][-1]) if self.training_metrics.get("loss_by_epoch") else float(self.stats.get("avg_loss", 0.0)),
            },
        )
        
        # Stop system monitoring
        self._stop_monitoring()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Save and upload datasets
        if self.config.save_datasets_locally or self.config.upload_datasets:
            self._save_and_upload_datasets(global_step)
        
        # Print comprehensive training summary
        self._print_training_summary()
        
        logger.info("Training completed!")
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics (CPU and memory usage)"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024 ** 3)
            memory_total_gb = memory.total / (1024 ** 3)
            memory_available_gb = memory.available / (1024 ** 3)
            
            # Process-specific memory
            process = psutil.Process()
            process_memory = process.memory_info()
            process_memory_gb = process_memory.rss / (1024 ** 3)
            
            # GPU/MPS memory and utilization (if using PyTorch)
            gpu_memory_used = 0.0
            gpu_memory_total = 0.0
            gpu_utilization = 0.0
            if torch.backends.mps.is_available():
                # MPS doesn't have direct memory query, but we can track allocations
                if hasattr(torch.mps, 'current_allocated_memory'):
                    gpu_memory_used = torch.mps.current_allocated_memory() / (1024 ** 3)
                    gpu_memory_total = torch.mps.driver_allocated_memory() / (1024 ** 3) if hasattr(torch.mps, 'driver_allocated_memory') else 0.0
                
                # For MPS, we estimate utilization based on memory usage and activity
                # MPS doesn't provide direct utilization metrics like CUDA
                # We use memory usage as a proxy: high memory usage = likely active
                if gpu_memory_total > 0:
                    memory_util = (gpu_memory_used / gpu_memory_total) * 100
                    # Also check if there are active operations (heuristic)
                    # If memory is being used, assume GPU is active
                    gpu_utilization = min(100.0, memory_util * 1.2)  # Scale up slightly as proxy
                else:
                    # Fallback: if we can't get memory, check if MPS is initialized
                    gpu_utilization = 50.0 if gpu_memory_used > 0 else 0.0
            elif torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / (1024 ** 3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                # CUDA provides utilization via nvidia-smi, but we can't query it directly
                # Use memory usage as proxy
                if gpu_memory_total > 0:
                    gpu_utilization = (gpu_memory_used / gpu_memory_total) * 100
                else:
                    gpu_utilization = 0.0
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_cores': len(cpu_per_core),
                'memory_percent': memory_percent,
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory_total_gb,
                'memory_available_gb': memory_available_gb,
                'process_memory_gb': process_memory_gb,
                'gpu_memory_used_gb': gpu_memory_used,
                'gpu_memory_total_gb': gpu_memory_total,
                'gpu_memory_percent': (gpu_memory_used / gpu_memory_total * 100) if gpu_memory_total > 0 else 0.0,
                'gpu_utilization': gpu_utilization,  # GPU/Neural Engine utilization estimate
            }
        except Exception as e:
            logger.warning(f"Error getting system metrics: {e}")
            return {}
    
    def _start_monitoring(self):
        """Start background thread for system monitoring"""
        if not self.monitoring_enabled or not self.writer:
            return
        
        def monitor_loop():
            step = 0
            while self.monitoring_enabled:
                try:
                    metrics = self._get_system_metrics()
                    if metrics and self.writer:
                        # Log CPU metrics
                        self.writer.add_scalar('System/CPU_Percent', metrics.get('cpu_percent', 0), step)
                        
                        # Log Memory metrics
                        self.writer.add_scalar('System/Memory_Percent', metrics.get('memory_percent', 0), step)
                        self.writer.add_scalar('System/Memory_Used_GB', metrics.get('memory_used_gb', 0), step)
                        self.writer.add_scalar('System/Memory_Available_GB', metrics.get('memory_available_gb', 0), step)
                        self.writer.add_scalar('System/Process_Memory_GB', metrics.get('process_memory_gb', 0), step)
                        
                        # Log GPU/MPS metrics if available
                        if metrics.get('gpu_memory_total_gb', 0) > 0:
                            self.writer.add_scalar('System/GPU_Memory_Used_GB', metrics.get('gpu_memory_used_gb', 0), step)
                            self.writer.add_scalar('System/GPU_Memory_Total_GB', metrics.get('gpu_memory_total_gb', 0), step)
                            self.writer.add_scalar('System/GPU_Memory_Percent', metrics.get('gpu_memory_percent', 0), step)
                            self.writer.add_scalar('System/GPU_Utilization', metrics.get('gpu_utilization', 0), step)
                    
                    step += 1
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    logger.warning(f"Error in monitoring loop: {e}")
                    time.sleep(self.monitoring_interval)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("System monitoring started")
    
    def _stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        logger.info("System monitoring stopped")
    
    def _print_training_summary(self):
        """Print comprehensive training summary with all metrics"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)
        
        # Calculate statistics
        gen_speeds = self.training_metrics['generation_tokens_per_sec']
        backprop_speeds = self.training_metrics['backprop_tokens_per_sec']
        
        # Generation performance
        if gen_speeds:
            avg_gen_speed = np.mean(gen_speeds)
            p99_gen_speed = np.percentile(gen_speeds, 99) if len(gen_speeds) > 0 else 0
            logger.info("\n📊 Generation Performance:")
            logger.info(f"  Average: {avg_gen_speed:.2f} tokens/sec")
            logger.info(f"  P99:     {p99_gen_speed:.2f} tokens/sec")
            logger.info(f"  Samples:  {len(gen_speeds)}")
        else:
            logger.info("\n📊 Generation Performance: No data")
        
        # Backpropagation performance
        if backprop_speeds:
            avg_backprop_speed = np.mean(backprop_speeds)
            p99_backprop_speed = np.percentile(backprop_speeds, 99) if len(backprop_speeds) > 0 else 0
            logger.info("\n🔄 Backpropagation Performance:")
            logger.info(f"  Average: {avg_backprop_speed:.2f} tokens/sec")
            logger.info(f"  P99:     {p99_backprop_speed:.2f} tokens/sec")
            logger.info(f"  Samples: {len(backprop_speeds)}")
        else:
            logger.info("\n🔄 Backpropagation Performance: No data")
        
        # API Token Usage
        total_api_tokens = self.training_metrics['api_tokens_sent']
        api_tokens_by_epoch = self.training_metrics['api_tokens_by_epoch']
        avg_tokens_per_epoch = np.mean(api_tokens_by_epoch) if api_tokens_by_epoch else 0
        
        logger.info("\n🌐 Teacher API Usage:")
        logger.info(f"  Total Tokens Sent: {total_api_tokens:,}")
        logger.info(f"  Average per Epoch: {avg_tokens_per_epoch:,.0f}")
        logger.info(f"  Breakdown by Epoch:")
        for i, tokens in enumerate(api_tokens_by_epoch, 1):
            logger.info(f"    Epoch {i}: {tokens:,} tokens")
        
        # Training Time
        if self.training_metrics['training_start_time'] and self.training_metrics['training_end_time']:
            total_time = self.training_metrics['training_end_time'] - self.training_metrics['training_start_time']
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            logger.info("\n⏱️  Training Duration:")
            logger.info(f"  Total Time: {hours}h {minutes}m {seconds}s ({total_time:.1f} seconds)")
        else:
            logger.info("\n⏱️  Training Duration: Not recorded")
        
        # Reward and Loss Trends
        reward_by_epoch = self.training_metrics['reward_by_epoch']
        loss_by_epoch = self.training_metrics['loss_by_epoch']
        reward_variance_by_epoch = self.training_metrics['reward_variance_by_epoch']
        
        if reward_by_epoch and len(reward_by_epoch) > 1:
            logger.info("\n📈 Training Trends:")
            logger.info(f"  {'Epoch':<8} {'Avg Reward':<15} {'Avg Loss':<15} {'Reward Variance':<18} {'Trend':<15}")
            logger.info("  " + "-"*75)
            for i, (reward, loss, variance) in enumerate(zip(reward_by_epoch, loss_by_epoch, reward_variance_by_epoch), 1):
                # Determine trends
                if i == 1:
                    reward_trend = "N/A"
                    loss_trend = "N/A"
                else:
                    reward_change = reward - reward_by_epoch[i-2]
                    loss_change = loss_by_epoch[i-2] - loss  # Loss should decrease
                    reward_trend = "↑" if reward_change > 0.01 else "↓" if reward_change < -0.01 else "→"
                    loss_trend = "↓" if loss_change > 0.01 else "↑" if loss_change < -0.01 else "→"
                
                logger.info(
                    f"  {i:<8} "
                    f"{reward:<15.4f} "
                    f"{loss:<15.4f} "
                    f"{variance:<18.4f} "
                    f"R:{reward_trend} L:{loss_trend}"
                )
            
            # Overall trend analysis
            if len(reward_by_epoch) >= 2:
                reward_trend_overall = reward_by_epoch[-1] - reward_by_epoch[0]
                loss_trend_overall = loss_by_epoch[0] - loss_by_epoch[-1]  # Loss should decrease
                avg_reward_variance = np.mean(reward_variance_by_epoch)
                final_reward = reward_by_epoch[-1]
                
                logger.info("\n📊 Trend Analysis:")
                logger.info(f"  Reward Change: {reward_trend_overall:+.4f} ({'↑ Improving' if reward_trend_overall > 0.01 else '↓ Declining' if reward_trend_overall < -0.01 else '→ Stable'})")
                logger.info(f"  Loss Change: {loss_trend_overall:+.4f} ({'↓ Improving' if loss_trend_overall > 0.01 else '↑ Worsening' if loss_trend_overall < -0.01 else '→ Stable'})")
                logger.info(f"  Avg Reward Variance: {avg_reward_variance:.4f} ({'✓ Low (consistent)' if avg_reward_variance < 0.01 else '⚠ High (inconsistent)'})")
                logger.info(f"  Final Reward: {final_reward:.4f} ({'✓ Target Met' if final_reward >= 0.7 else '⚠ Below Target' if final_reward >= 0.5 else '✗ Needs Improvement'})")
                
                # Convergence check
                if len(reward_by_epoch) >= 3:
                    recent_rewards = reward_by_epoch[-3:]
                    reward_std = np.std(recent_rewards)
                    if reward_std < 0.02 and final_reward >= 0.65:
                        logger.info(f"  Convergence: ✓ Converged (reward stable at {final_reward:.4f})")
                    elif reward_std < 0.02:
                        logger.info(f"  Convergence: → Stable but below target (reward: {final_reward:.4f})")
                    else:
                        logger.info(f"  Convergence: ⚠ Still learning (reward variance: {reward_std:.4f})")
        else:
            logger.info("\n📈 Training Trends: Insufficient data (need at least 2 epochs)")
        
        # Scoring Breakdown by Epoch
        scoring_breakdown = self.training_metrics['scoring_breakdown_by_epoch']
        if scoring_breakdown:
            logger.info("\n📈 Scoring Breakdown Trend by Epoch:")
            logger.info("  (Weights: Correctness=0.3, Code Quality=0.3, Efficiency=0.2, Documentation=0.2)")
            logger.info(f"  {'Epoch':<8} {'Correctness':<15} {'Code Quality':<15} {'Efficiency':<15} {'Documentation':<15}")
            logger.info("  " + "-"*70)
            for i, breakdown in enumerate(scoring_breakdown, 1):
                logger.info(
                    f"  {i:<8} "
                    f"{breakdown.get('correctness', 0.0):<15.3f} "
                    f"{breakdown.get('code_quality', 0.0):<15.3f} "
                    f"{breakdown.get('efficiency', 0.0):<15.3f} "
                    f"{breakdown.get('documentation', 0.0):<15.3f}"
                )
        else:
            logger.info("\n📈 Scoring Breakdown Trend: No data")
        
        logger.info("\n" + "="*80 + "\n")
    
    def _log_stats(self, step: int, loss_dict: Dict, rewards: List[float]):
        """Log training statistics"""
        avg_reward = np.mean(rewards) if rewards else 0.0
        
        if self.writer:
            # Training metrics
            self.writer.add_scalar('Train/Loss', loss_dict['loss'], step)
            self.writer.add_scalar('Train/PolicyLoss', loss_dict['policy_loss'], step)
            self.writer.add_scalar('Train/KLPenalty', loss_dict['kl_penalty'], step)
            self.writer.add_scalar('Train/AvgReward', avg_reward, step)
            
            # Reward statistics
            if rewards and len(rewards) > 1:
                self.writer.add_scalar('Train/RewardStd', np.std(rewards), step)
                self.writer.add_scalar('Train/RewardMin', np.min(rewards), step)
                self.writer.add_scalar('Train/RewardMax', np.max(rewards), step)
                self.writer.add_scalar('Train/RewardVariance', np.var(rewards), step)
            
            # Performance metrics (latest values)
            gen_speeds = self.training_metrics['generation_tokens_per_sec']
            backprop_speeds = self.training_metrics['backprop_tokens_per_sec']
            if gen_speeds:
                self.writer.add_scalar('Performance/Generation_TokensPerSec', gen_speeds[-1], step)
            if backprop_speeds:
                self.writer.add_scalar('Performance/Backprop_TokensPerSec', backprop_speeds[-1], step)
        
        # Get system metrics for this step
        system_metrics = self._get_system_metrics()
        
        stats_str = (
            f"Step {step}:\n"
            f"  Loss: {loss_dict['loss']:.4f}\n"
            f"  Policy Loss: {loss_dict['policy_loss']:.4f}\n"
            f"  KL Penalty: {loss_dict['kl_penalty']:.4f}\n"
            f"  Avg Reward: {avg_reward:.4f}\n"
            f"  Reward Std: {np.std(rewards):.4f}" if rewards else ""
        )
        
        # Add system metrics to log string
        if system_metrics:
            stats_str += (
                f"\n  CPU: {system_metrics.get('cpu_percent', 0):.1f}%\n"
                f"  Memory: {system_metrics.get('memory_percent', 0):.1f}% "
                f"({system_metrics.get('memory_used_gb', 0):.2f}GB / {system_metrics.get('memory_total_gb', 0):.2f}GB)\n"
                f"  Process Memory: {system_metrics.get('process_memory_gb', 0):.2f}GB"
            )
            if system_metrics.get('gpu_memory_total_gb', 0) > 0:
                gpu_util = system_metrics.get('gpu_utilization', 0)
                stats_str += (
                    f"\n  GPU Memory: {system_metrics.get('gpu_memory_percent', 0):.1f}% "
                    f"({system_metrics.get('gpu_memory_used_gb', 0):.2f}GB / {system_metrics.get('gpu_memory_total_gb', 0):.2f}GB)\n"
                    f"  GPU Utilization: {gpu_util:.1f}%"
                )
        
        logger.info(stats_str)
        
        # System metrics at logging steps (already logged above, but ensure they're in TensorBoard)
        if self.writer and system_metrics:
            self.writer.add_scalar('System/CPU_Percent', system_metrics.get('cpu_percent', 0), step)
            self.writer.add_scalar('System/Memory_Percent', system_metrics.get('memory_percent', 0), step)
            self.writer.add_scalar('System/Memory_Used_GB', system_metrics.get('memory_used_gb', 0), step)
            self.writer.add_scalar('System/Process_Memory_GB', system_metrics.get('process_memory_gb', 0), step)
            if system_metrics.get('gpu_memory_total_gb', 0) > 0:
                self.writer.add_scalar('System/GPU_Memory_Used_GB', system_metrics.get('gpu_memory_used_gb', 0), step)
                self.writer.add_scalar('System/GPU_Memory_Percent', system_metrics.get('gpu_memory_percent', 0), step)
                self.writer.add_scalar('System/GPU_Utilization', system_metrics.get('gpu_utilization', 0), step)
    
    def _save_checkpoint(self, step: int, final: bool = False, checkpoint_name: Optional[str] = None, summary: Optional[Dict] = None):
        """Save model checkpoint in both PyTorch and MLX formats"""
        # IMPORTANT: step can be 0 for long periods with large grad accumulation.
        # Use a unique checkpoint_name when provided to avoid overwriting checkpoint-0.
        name = checkpoint_name or f"checkpoint-{step}"
        checkpoint_dir = Path(self.config.output_dir) / name
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            # Avoid overwriting an existing checkpoint directory.
            suffix = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            checkpoint_dir = Path(self.config.output_dir) / f"{name}-{suffix}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch format (standard)
        logger.info(f"Saving PyTorch checkpoint to {checkpoint_dir}")
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save MLX format if enabled
        mlx_dir = None
        if self.config.save_mlx_format:
            try:
                mlx_dir = self._save_mlx_checkpoint(checkpoint_dir, step)
            except Exception as e:
                logger.warning(f"Failed to save MLX checkpoint: {e}. Continuing with PyTorch format only.")
        
        # Upload to Hugging Face if enabled (only on final checkpoint or if specified)
        if self.config.upload_to_hub and (final or step % (self.config.save_steps * 2) == 0):
            try:
                if mlx_dir:
                    self._upload_to_huggingface(mlx_dir, step, final)
                else:
                    logger.warning("MLX model not available for upload. Skipping HF upload.")
            except Exception as e:
                logger.warning(f"Failed to upload to Hugging Face: {e}")
        
        # Save training stats
        stats_file = checkpoint_dir / "training_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

        # Save checkpoint summary (epoch/batch context + baseline gain)
        if summary:
            try:
                summary_payload = dict(summary)
                summary_payload.setdefault("checkpoint_name", checkpoint_dir.name)
                summary_payload.setdefault("step", int(step))
                summary_payload.setdefault("final", bool(final))
                if self.baseline_reward is not None:
                    summary_payload.setdefault("baseline_reward", float(self.baseline_reward))
                    if "avg_reward" in summary_payload:
                        summary_payload.setdefault("reward_gain_from_baseline", float(summary_payload["avg_reward"]) - float(self.baseline_reward))
                summary_payload.setdefault("ts_iso", datetime.utcnow().isoformat() + "Z")
                with open(checkpoint_dir / "checkpoint_summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary_payload, f, indent=2)
            except Exception as e:
                logger.debug(f"Failed to write checkpoint_summary.json: {e}")

        # Keep only the most recent checkpoints (best-effort)
        try:
            limit = int(getattr(self.config, "save_total_limit", 0) or 0)
            if limit > 0:
                ckpt_root = Path(self.config.output_dir)
                ckpts = [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
                ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                for old in ckpts[limit:]:
                    try:
                        import shutil
                        shutil.rmtree(old)
                    except Exception:
                        pass
        except Exception:
            pass
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def _save_mlx_checkpoint(self, checkpoint_dir: Path, step: int):
        """Convert and save model to MLX format for faster inference on Apple Silicon"""
        try:
            import mlx.core as mx
            from mlx_lm import convert, quantize
            from mlx_lm.utils import fetch_from_hub
        except ImportError:
            logger.warning("MLX libraries not available. Install with: pip install mlx mlx-lm")
            return
        
        mlx_dir = checkpoint_dir / "mlx_model"
        mlx_dir.mkdir(exist_ok=True)
        
        logger.info(f"Converting model to MLX format at {mlx_dir}")
        
        # Get the model path (either local checkpoint or original base model)
        model_path = str(checkpoint_dir)
        
        # Convert the model to MLX format
        # Note: MLX conversion works best with HuggingFace models
        # We'll save the model first, then convert it
        try:
            # Convert model weights to MLX format
            # This uses mlx-lm's convert utility which handles the conversion
            logger.info("Converting PyTorch weights to MLX format...")
            
            # Save model config for MLX
            import shutil
            config_file = checkpoint_dir / "config.json"
            if config_file.exists():
                shutil.copy(config_file, mlx_dir / "config.json")
            
            # Copy tokenizer files
            tokenizer_files = ['tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json']
            for file in tokenizer_files:
                src_file = checkpoint_dir / file
                if src_file.exists():
                    shutil.copy(src_file, mlx_dir / file)
            
            # Convert model weights
            # MLX uses safetensors format
            self._convert_weights_to_mlx(checkpoint_dir, mlx_dir)
            
            # Apply quantization if specified
            if self.config.mlx_quantization:
                logger.info(f"Applying {self.config.mlx_quantization} quantization to MLX model...")
                self._quantize_mlx_model(mlx_dir, self.config.mlx_quantization)
            
            logger.info(f"MLX model saved to {mlx_dir}")
            
            # Create a README for MLX usage
            mlx_readme = mlx_dir / "README_MLX.md"
            with open(mlx_readme, 'w') as f:
                f.write(f"""# MLX Model Checkpoint

This directory contains the model in MLX format, optimized for Apple Silicon (M5 MacBook).

## Usage

```python
from mlx_lm import load, generate

# Load the model
model, tokenizer = load("{mlx_dir}")

# Generate text
prompt = "Write high-quality python code:"
response = generate(model, tokenizer, prompt=prompt, max_tokens=512)
print(response)
```

## Quantization

{"Quantized with " + self.config.mlx_quantization if self.config.mlx_quantization else "No quantization applied"}

## Conversion Info

- Converted from PyTorch checkpoint: checkpoint-{step}
- Original model: {self.config.base_model}
- Conversion date: {datetime.now().isoformat()}
""")
            
            # Return mlx_dir for potential upload
            return mlx_dir
        
        except Exception as e:
            logger.error(f"Error during MLX conversion: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise
    
    def _convert_weights_to_mlx(self, pytorch_dir: Path, mlx_dir: Path):
        """Convert PyTorch model weights to MLX safetensors format"""
        try:
            # Try using mlx-lm's convert utility first (preferred method)
            from mlx_lm import convert
            
            logger.info("Using mlx-lm convert utility for model conversion...")
            pytorch_path = str(pytorch_dir.absolute())
            mlx_path = str(mlx_dir.absolute())
            
            # Convert the model
            convert(pytorch_path, mlx_path)
            logger.info(f"Successfully converted model to MLX format using mlx-lm")
            return
        
        except ImportError:
            logger.warning("mlx-lm not available. Attempting manual conversion...")
        except Exception as e:
            logger.warning(f"mlx-lm convert failed: {e}. Attempting manual conversion...")
        
        # Fallback: Manual conversion
        try:
            import torch
            from safetensors.numpy import save_file as save_numpy_safetensors
            
            # Load PyTorch model weights
            pytorch_model_file = pytorch_dir / "pytorch_model.bin"
            if not pytorch_model_file.exists():
                # Try safetensors format
                pytorch_model_file = pytorch_dir / "model.safetensors"
            
            if not pytorch_model_file.exists():
                logger.warning("No PyTorch model file found. Skipping MLX conversion.")
                return
            
            # Load the state dict
            if pytorch_model_file.suffix == '.safetensors':
                from safetensors import safe_open
                state_dict = {}
                with safe_open(pytorch_model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(pytorch_model_file, map_location="cpu")
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
            
            # Convert to MLX format (numpy arrays)
            mlx_state_dict = {}
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    # Convert to numpy
                    np_array = value.detach().cpu().numpy().astype(np.float32)
                    mlx_state_dict[key] = np_array
            
            # Save as safetensors for MLX
            mlx_model_file = mlx_dir / "model.safetensors"
            save_numpy_safetensors(mlx_state_dict, mlx_model_file)
            
            logger.info(f"Manually converted {len(mlx_state_dict)} weight tensors to MLX format")
        
        except Exception as e:
            logger.error(f"Manual MLX conversion also failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise
    
    def _quantize_mlx_model(self, mlx_dir: Path, quantization: str):
        """Apply quantization to MLX model"""
        try:
            from mlx_lm import quantize
            
            if quantization == "q4_bit":
                bits = 4
            elif quantization == "q8_bit":
                bits = 8
            else:
                logger.warning(f"Unknown quantization: {quantization}")
                return
            
            logger.info(f"Quantizing MLX model to {bits}-bit...")
            # Note: mlx_lm.quantize typically works on model loading
            # We'll create a quantized version
            quantized_dir = mlx_dir.parent / f"{mlx_dir.name}_quantized_{bits}bit"
            quantized_dir.mkdir(exist_ok=True)
            
            # The actual quantization happens during model loading in MLX
            # We'll just note it in the config
            logger.info(f"Quantization will be applied when loading the model with mlx_lm.load()")
            
        except Exception as e:
            logger.warning(f"Could not quantize MLX model: {e}")
    
    def _upload_to_huggingface(self, mlx_dir: Path, step: int, final: bool = False):
        """Upload MLX model to Hugging Face MLX Community"""
        if not self.config.hf_repo_id:
            logger.warning("HF repo_id not specified. Skipping upload.")
            return
        
        try:
            from huggingface_hub import HfApi, create_repo, upload_folder
            from huggingface_hub.utils import HfHubHTTPError
            
            hf_token = os.getenv(self.config.hf_token_env)
            if not hf_token:
                logger.warning(f"Hugging Face token not found in {self.config.hf_token_env}. Skipping upload.")
                return
            
            api = HfApi(token=hf_token)
            repo_id = self.config.hf_repo_id
            
            # Create repository if it doesn't exist
            try:
                create_repo(
                    repo_id=repo_id,
                    token=hf_token,
                    private=self.config.hf_private,
                    repo_type="model",
                    exist_ok=True
                )
                logger.info(f"Repository {repo_id} ready")
            except HfHubHTTPError as e:
                logger.warning(f"Could not create repository: {e}")
                return
            
            # Upload MLX model
            logger.info(f"Uploading MLX model to {repo_id}...")
            
            # Create model card
            model_card = f"""---
library_name: mlx
tags:
- code-generation
- python
- cpp
- rust
- rlaif
- qwen
- fine-tuned
base_model: {self.config.base_model}
---

# {repo_id.split('/')[-1]}

This model is a fine-tuned version of [{self.config.base_model}](https://huggingface.co/{self.config.base_model}) using Reinforcement Learning from AI Feedback (RLAIF).

## Training Details

- **Base Model**: {self.config.base_model}
- **Training Method**: RLAIF (Reinforcement Learning from AI Feedback)
- **Teacher Model**: {self.config.teacher_provider}/{self.config.teacher_model}
- **Training Steps**: {step}
- **Languages**: Python, C++, Rust
- **Format**: MLX (optimized for Apple Silicon)

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("{repo_id}")
response = generate(model, tokenizer, prompt="Write high-quality python code:\\n\\nImplement binary search\\n\\nCode:", max_tokens=512)
```

## Training Statistics

- Average Reward: {self.stats.get('avg_reward', 0.0):.4f}
- Average Loss: {self.stats.get('avg_loss', 0.0):.4f}
- Total Samples: {self.stats.get('num_samples', 0)}

## Model Card

This model was trained to generate high-quality code in Python, C++, and Rust using a teacher-student training scheme where a teacher model (OpenAI/Claude) provides feedback to improve code quality.
"""
            
            # Save model card
            readme_path = mlx_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write(model_card)
            
            # Upload folder
            upload_folder(
                folder_path=str(mlx_dir),
                repo_id=repo_id,
                token=hf_token,
                commit_message=f"Upload checkpoint-{step} (MLX format)" + (" [FINAL]" if final else ""),
                ignore_patterns=["*.bin", "*.pt", "*.pth"]  # Don't upload PyTorch files
            )
            
            logger.info(f"✓ Successfully uploaded model to https://huggingface.co/{repo_id}")
            
            # If upload_quantized is enabled, also upload quantized version
            if self.config.upload_quantized and self.config.mlx_quantization:
                quantized_repo_id = f"{repo_id}-{self.config.mlx_quantization}"
                logger.info(f"Uploading quantized version to {quantized_repo_id}...")
                # Note: Quantization upload would require creating quantized model first
                # This is a placeholder for future implementation
                
        except ImportError:
            logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
        except Exception as e:
            logger.error(f"Error uploading to Hugging Face: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise
    
    def _save_and_upload_datasets(self, step: int):
        """Save datasets locally and upload to Hugging Face"""
        if not self.dataset_collection['training']:
            logger.warning("No dataset entries collected. Skipping dataset save/upload.")
            return
        
        dataset_dir = Path(self.config.dataset_output_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Saving datasets locally...")
        
        # Save training dataset
        if self.dataset_collection['training']:
            train_file = dataset_dir / "train.jsonl"
            with open(train_file, 'w') as f:
                for entry in self.dataset_collection['training']:
                    f.write(json.dumps(entry) + '\n')
            logger.info(f"Saved {len(self.dataset_collection['training'])} training examples to {train_file}")
        
        # Save validation dataset
        if self.dataset_collection['validation']:
            val_file = dataset_dir / "validation.jsonl"
            with open(val_file, 'w') as f:
                for entry in self.dataset_collection['validation']:
                    f.write(json.dumps(entry) + '\n')
            logger.info(f"Saved {len(self.dataset_collection['validation'])} validation examples to {val_file}")
        
        # Save evaluation dataset
        if self.dataset_collection['evaluation']:
            eval_file = dataset_dir / "evaluation.jsonl"
            with open(eval_file, 'w') as f:
                for entry in self.dataset_collection['evaluation']:
                    f.write(json.dumps(entry) + '\n')
            logger.info(f"Saved {len(self.dataset_collection['evaluation'])} evaluation examples to {eval_file}")
        
        # Create dataset card
        dataset_card = self._create_dataset_card(step)
        readme_file = dataset_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(dataset_card)
        
        # Upload to Hugging Face if enabled
        if self.config.upload_datasets and self.config.dataset_repo_id:
            self._upload_dataset_to_hf(dataset_dir, step)
    
    def _create_dataset_card(self, step: int) -> str:
        """Create a dataset card for Hugging Face"""
        num_train = len(self.dataset_collection['training'])
        num_val = len(self.dataset_collection['validation'])
        num_eval = len(self.dataset_collection['evaluation'])
        
        card = f"""---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- code-generation
- python
- cpp
- rust
- rlaif
- reinforcement-learning
size_categories:
- 1K<n<10K
---

# Code RLAIF Dataset

This dataset contains prompts, teacher-generated code, student-generated code, and scoring parameters from the RLAIF (Reinforcement Learning from AI Feedback) training process.

## Dataset Description

This dataset was generated during the fine-tuning of a Qwen model for code generation using RLAIF methodology. Each entry includes:

- **Prompt**: The original code generation prompt
- **Language**: Programming language (Python, C++, or Rust)
- **Student Code**: Code generated by the student model (Qwen)
- **Teacher Code**: High-quality reference code generated by the teacher model (OpenAI/Claude)
- **Student Score**: Quality score for student code (0.0-1.0)
- **Teacher Score**: Quality score for teacher code (0.0-1.0)
- **Reward**: Normalized reward (student_score / teacher_score)
- **Scoring Breakdown**: Detailed scoring parameters
- **Timestamp**: When the entry was created

## Dataset Structure

- **Training Set**: {num_train} examples
- **Validation Set**: {num_val} examples
- **Evaluation Set**: {num_eval} examples

## Data Fields

Each example contains:
- `prompt` (string): Code generation prompt
- `language` (string): Programming language (python, cpp, rust)
- `student_code` (string): Code generated by student model
- `teacher_code` (string): Reference code from teacher model
- `student_score` (float): Quality score for student code
- `teacher_score` (float): Quality score for teacher code
- `reward` (float): Normalized reward value
- `scoring_breakdown` (dict): Detailed scoring parameters
- `timestamp` (string): ISO timestamp

## Usage

### Load from Hugging Face

```python
from datasets import load_dataset

dataset = load_dataset("{self.config.dataset_repo_id or 'mlx-community/code-rlaif-dataset'}")

# Access training data
train_data = dataset['train']
print(train_data[0])
```

### Load from Local Files

```python
import json

# Load training data
with open('datasets/train.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line)
        print(entry['prompt'])
        print(entry['teacher_code'])
        print(f"Score: {{entry['student_score']}}")
```

## Training Details

- **Base Model**: {self.config.base_model}
- **Teacher Model**: {self.config.teacher_provider}/{self.config.teacher_model}
- **Training Steps**: {step}
- **Languages**: Python, C++, Rust
- **Average Reward**: {self.stats.get('avg_reward', 0.0):.4f}

## Scoring Methodology

Scores are computed using the teacher model evaluating code on:
- **Correctness** (30%): Does the code solve the problem correctly?
- **Code Quality** (30%): Is it clean, readable, and well-structured?
- **Efficiency** (20%): Is it efficient and follows best practices?
- **Documentation** (20%): Is it well-documented?

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{code_rlaif_dataset,
  title={{Code RLAIF Dataset}},
  author={{MLX Community}},
  year={{2024}},
  url={{https://huggingface.co/datasets/{self.config.dataset_repo_id or 'mlx-community/code-rlaif-dataset'}}}
}}
```

## License

MIT License
"""
        return card
    
    def _upload_dataset_to_hf(self, dataset_dir: Path, step: int):
        """Upload dataset to Hugging Face"""
        if not self.config.dataset_repo_id:
            logger.warning("Dataset repo_id not specified. Skipping upload.")
            return
        
        try:
            from huggingface_hub import HfApi, create_repo, upload_folder
            from huggingface_hub.utils import HfHubHTTPError
            
            hf_token = os.getenv(self.config.hf_token_env)
            if not hf_token:
                logger.warning(f"Hugging Face token not found in {self.config.hf_token_env}. Skipping dataset upload.")
                return
            
            repo_id = self.config.dataset_repo_id
            
            # Create dataset repository if it doesn't exist
            try:
                create_repo(
                    repo_id=repo_id,
                    token=hf_token,
                    private=self.config.hf_private,
                    repo_type="dataset",
                    exist_ok=True
                )
                logger.info(f"Dataset repository {repo_id} ready")
            except HfHubHTTPError as e:
                logger.warning(f"Could not create dataset repository: {e}")
                return
            
            # Upload dataset
            logger.info(f"Uploading dataset to {repo_id}...")
            
            upload_folder(
                folder_path=str(dataset_dir),
                repo_id=repo_id,
                token=hf_token,
                commit_message=f"Upload RLAIF training dataset (step {step})",
                ignore_patterns=["*.pyc", "__pycache__"]
            )
            
            logger.info(f"✓ Successfully uploaded dataset to https://huggingface.co/datasets/{repo_id}")
            
        except ImportError:
            logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
        except Exception as e:
            logger.error(f"Error uploading dataset to Hugging Face: {e}")
            import traceback
            logger.debug(traceback.format_exc())


def load_config(config_path: str) -> Tuple[RLAIFConfig, dict]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Flatten nested config
    model_cfg = config_dict.get('model', {})
    teacher_cfg = config_dict.get('teacher', {})
    training_cfg = config_dict.get('training', {})
    rlaif_cfg = config_dict.get('rlaif', {})
    logging_cfg = config_dict.get('logging', {})
    hardware_cfg = config_dict.get('hardware', {})
    data_cfg = config_dict.get('data', {})
    
    # Helper function to ensure proper type conversion
    def to_int(value, default):
        if value is None:
            return default
        return int(value)
    
    def to_float(value, default):
        if value is None:
            return default
        return float(value)
    
    def to_bool(value, default):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    
    rlaif_config = RLAIFConfig(
        base_model=model_cfg.get('base_model', 'Qwen/Qwen2.5-Coder-3B-Instruct'),
        teacher_provider=teacher_cfg.get('provider', 'openai'),
        teacher_model=teacher_cfg.get('model_name', 'claude-3-5-haiku-20241022' if teacher_cfg.get('provider') == 'anthropic' else 'gpt-4-turbo-preview'),
        teacher_api_key_env=teacher_cfg.get('api_key_env', 'OPENAI_API_KEY'),
        output_dir=training_cfg.get('output_dir', './checkpoints'),
        num_epochs=to_int(training_cfg.get('num_epochs'), 3),
        batch_size=to_int(training_cfg.get('batch_size'), 4),
        gradient_accumulation_steps=to_int(training_cfg.get('gradient_accumulation_steps'), 8),
        learning_rate=to_float(training_cfg.get('learning_rate'), 2e-5),
        warmup_steps=to_int(training_cfg.get('warmup_steps'), 100),
        save_steps=to_int(training_cfg.get('save_steps'), 500),
        eval_steps=to_int(training_cfg.get('eval_steps'), 250),
        logging_steps=to_int(training_cfg.get('logging_steps'), 50),
        save_every_epochs=to_int(training_cfg.get('save_every_epochs'), 1),
        save_every_batches=to_int(training_cfg.get('save_every_batches'), 0),
        max_grad_norm=to_float(training_cfg.get('max_grad_norm'), 1.0),
        weight_decay=to_float(training_cfg.get('weight_decay'), 0.01),
        lr_scheduler_type=training_cfg.get('lr_scheduler_type', 'cosine'),
        reward_weight=to_float(rlaif_cfg.get('reward_weight'), 1.0),
        kl_penalty=to_float(rlaif_cfg.get('kl_penalty'), 0.1),
        beta=to_float(rlaif_cfg.get('beta'), 0.1),
        num_samples_per_prompt=to_int(rlaif_cfg.get('num_samples_per_prompt'), 4),
        generation_temperature=to_float(rlaif_cfg.get('generation_temperature'), 0.8),
        curriculum_learning=to_bool(rlaif_cfg.get('curriculum_learning'), False),
        reward_bonuses=to_bool(rlaif_cfg.get('reward_bonuses'), False),
        use_lora=to_bool(rlaif_cfg.get('use_lora'), False),
        use_qlora=to_bool(rlaif_cfg.get('use_qlora'), False),
        lora_r=to_int(rlaif_cfg.get('lora_r'), 16),
        lora_alpha=to_int(rlaif_cfg.get('lora_alpha'), 32),
        lora_dropout=to_float(rlaif_cfg.get('lora_dropout'), 0.05),
        lora_target_modules=rlaif_cfg.get('lora_target_modules'),  # Can be None or list
        max_length=to_int(model_cfg.get('max_length'), 2048),
        use_4bit=to_bool(model_cfg.get('use_4bit'), True),
        use_mps=to_bool(hardware_cfg.get('use_mps'), True),
        mixed_precision=hardware_cfg.get('mixed_precision', 'bf16'),
        tensorboard_dir=logging_cfg.get('tensorboard_dir', './logs/tensorboard'),
        log_level=logging_cfg.get('log_level', 'INFO'),
        save_json_summaries=to_bool(logging_cfg.get('save_json_summaries'), True),
        json_summaries_dir=logging_cfg.get('json_summaries_dir', './logs/json_summaries'),
        baseline_eval_batches=to_int(logging_cfg.get('baseline_eval_batches'), 1),
        tensorboard_batch_interval=to_int(logging_cfg.get('tensorboard_batch_interval'), 1),
        top_k=to_int(rlaif_cfg.get('top_k'), 50),
        top_p=to_float(rlaif_cfg.get('top_p'), 0.95),
        save_mlx_format=to_bool(hardware_cfg.get('save_mlx_format'), True),
        mlx_quantization=hardware_cfg.get('mlx_quantization', None),
        use_safetensors=to_bool(model_cfg.get('use_safetensors'), True),
        low_cpu_mem_usage=to_bool(model_cfg.get('low_cpu_mem_usage'), True),
        upload_to_hub=to_bool(config_dict.get('huggingface', {}).get('upload_to_hub'), False),
        hf_repo_id=config_dict.get('huggingface', {}).get('repo_id', None),
        hf_token_env=config_dict.get('huggingface', {}).get('hf_token_env', 'HUGGINGFACE_TOKEN'),
        upload_quantized=to_bool(config_dict.get('huggingface', {}).get('upload_quantized'), True),
        hf_private=to_bool(config_dict.get('huggingface', {}).get('private'), False),
        upload_datasets=to_bool(config_dict.get('huggingface', {}).get('upload_datasets'), True),
        dataset_repo_id=config_dict.get('huggingface', {}).get('dataset_repo_id', None),
        save_datasets_locally=to_bool(config_dict.get('huggingface', {}).get('save_datasets_locally'), True),
        dataset_output_dir=config_dict.get('huggingface', {}).get('dataset_output_dir', './datasets'),
        use_mlx_for_generation=to_bool(hardware_cfg.get('use_mlx_for_generation'), True),  # Default to True for better performance
        mlx_model_path=hardware_cfg.get('mlx_model_path', None),
        use_unsloth=to_bool(hardware_cfg.get('use_unsloth'), False),
        unsloth_dtype=hardware_cfg.get('unsloth_dtype', 'bf16'),
        unsloth_max_seq_length=to_int(hardware_cfg.get('unsloth_max_seq_length'), None),
    )
    
    return rlaif_config, data_cfg


def main():
    parser = argparse.ArgumentParser(
        description="RLAIF Training for Code Generation Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model (Qwen2.5-Coder-3B-Instruct)
  python train_rlaif.py --config config.yaml

  # Use a different model
  python train_rlaif.py --config config.yaml --model Qwen/Qwen2.5-7B-Instruct

  # Use a local model
  python train_rlaif.py --config config.yaml --model ./my_local_model

  # Custom data files
  python train_rlaif.py --config config.yaml --train_file ./data/train.jsonl
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model name or path (overrides config, e.g., Qwen/Qwen2.5-Coder-3B-Instruct)'
    )
    parser.add_argument(
        '--train_file',
        type=str,
        help='Path to training data file (overrides config)'
    )
    parser.add_argument(
        '--eval_file',
        type=str,
        help='Path to evaluation data file (overrides config)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable DEBUG logging (including HTTP request logs)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config, data_cfg = load_config(args.config)
    
    # Override model if provided via command line
    if args.model:
        config.base_model = args.model
        logger.info(f"Using model from command line: {config.base_model}")

    # Enable debug logging if requested
    if args.debug:
        config.log_level = "DEBUG"
    
    # Setup logging level
    logging.getLogger().setLevel(getattr(logging, config.log_level))
    
    # Suppress httpx HTTP request logs unless in DEBUG mode
    # httpx logs every HTTP request at INFO level, which is too verbose for training
    httpx_logger = logging.getLogger("httpx")
    if config.log_level.upper() == "DEBUG":
        httpx_logger.setLevel(logging.INFO)  # Show HTTP requests in DEBUG mode
    else:
        httpx_logger.setLevel(logging.WARNING)  # Suppress HTTP requests in INFO/WARNING mode
    
    # Check for API key before proceeding
    api_key_env = config.teacher_api_key_env
    api_key = os.getenv(api_key_env)
    if not api_key:
        # Check if the other provider's key is set as a helpful hint
        other_key_env = "ANTHROPIC_API_KEY" if api_key_env == "OPENAI_API_KEY" else "OPENAI_API_KEY"
        other_key = os.getenv(other_key_env)
        
        error_msg = (
            f"\n{'='*80}\n"
            f"ERROR: API key not found in environment variable '{api_key_env}'\n"
            f"{'='*80}\n"
        )
        
        if other_key:
            error_msg += (
                f"Note: Found {other_key_env} but config requires {api_key_env}.\n"
                f"Either:\n"
                f"  1. Set {api_key_env}, or\n"
                f"  2. Update config.yaml to use provider '{'anthropic' if api_key_env == 'OPENAI_API_KEY' else 'openai'}'\n\n"
            )
        
        error_msg += (
            f"Please set your API key:\n"
            f"  export {api_key_env}='your-api-key'\n\n"
            f"For OpenAI:\n"
            f"  export OPENAI_API_KEY='sk-...'\n"
            f"  Get your key from: https://platform.openai.com/api-keys\n\n"
            f"For Anthropic:\n"
            f"  export ANTHROPIC_API_KEY='sk-ant-...'\n"
            f"  Get your key from: https://console.anthropic.com/\n\n"
            f"Or use the run_training.sh script which checks for API keys.\n"
            f"{'='*80}\n"
        )
        logger.error(error_msg)
        return 1
    
    # Load datasets
    train_file = args.train_file or data_cfg.get('train_file', './data/train.jsonl')
    eval_file = args.eval_file or data_cfg.get('eval_file', './data/eval.jsonl')
    
    # Initialize trainer
    try:
        trainer = RLAIFTrainer(config)
    except ValueError as e:
        logger.error(str(e))
        return 1
    
    # Load training dataset
    train_dataset = CodeDataset(train_file, trainer.tokenizer, config.max_length)
    
    # Load eval dataset if provided
    eval_dataset = None
    if os.path.exists(eval_file):
        eval_dataset = CodeDataset(eval_file, trainer.tokenizer, config.max_length)
    
    # Start training
    try:
        trainer.train(train_dataset, eval_dataset)
        logger.info("Training completed successfully!")
        return 0
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

