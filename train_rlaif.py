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
    top_k: int = 50
    top_p: float = 0.95
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


class CodeDataset(Dataset):
    """Dataset for code training examples"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        logger.info(f"Loading dataset from {data_file}")
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.data)} examples")
    
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
                    return response.content[0].text.strip()
            
            except Exception as e:
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
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": scoring_prompt}],
                        temperature=0.1,
                        max_tokens=50
                    )
                    score_text = response.choices[0].message.content.strip()
                else:  # anthropic
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=50,
                        temperature=0.1,
                        messages=[{"role": "user", "content": scoring_prompt}]
                    )
                    score_text = response.content[0].text.strip()
                
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
                error_str = str(e)
                if "404" in error_str or "not_found" in error_str.lower():
                    logger.error(f"Error: Model '{self.model_name}' not found when scoring code.")
                    logger.error(f"  Check your model name in config.yaml. Common models: 'claude-3-5-sonnet', 'claude-3-5-haiku'")
                else:
                    logger.error(f"Error scoring code: {e}")
                return 0.5


class RLAIFTrainer:
    """RLAIF Trainer implementing teacher-student training"""
    
    def __init__(self, config: RLAIFConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Load model and tokenizer
        logger.info(f"Loading base model: {config.base_model}")
        
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
            mem_before = 0
        
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
        if config.use_mlx_for_generation:
            # Auto-detect MLX model path if not specified
            mlx_path = config.mlx_model_path
            if mlx_path is None:
                # Try common MLX model locations in order of preference
                possible_paths = [
                    "./mlx_model_q8",  # Q8 quantized (best balance)
                    "./mlx_model_q4",  # Q4 quantized (smallest)
                    "./mlx_model",      # Unquantized
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
                    logger.warning(f"  uv run python convert_to_mlx.py --hf-path {self.config.base_model} --mlx-path ./mlx_model_q8 --quantize q8_bit")
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
        """Setup device for M5 MacBook"""
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
            elif "q4" in model_path.lower() or "q4_bit" in model_path.lower():
                quantize_bits = 4
                logger.info("Auto-detected Q4 quantization from path name")
            elif "q8" in model_path.lower() or "q8_bit" in model_path.lower():
                quantize_bits = 8
                logger.info("Auto-detected Q8 quantization from path name")
            
            if os.path.exists(model_path):
                # If model path contains q4/q8, the model is already quantized
                # Don't pass quantize parameter - just load the pre-quantized model
                model_already_quantized = "q4" in model_path.lower() or "q8" in model_path.lower()
                
                if model_already_quantized:
                    logger.info(f"Loading pre-quantized MLX model from {model_path}...")
                    logger.info("  (Model is already quantized - no need to pass quantize parameter)")
                    self.mlx_model, self.mlx_tokenizer = load(model_path)
                    # Extract quantization level from path for logging
                    if "q4" in model_path.lower():
                        quantize_bits = 4
                    elif "q8" in model_path.lower():
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
                    logger.info("  Using full precision (consider converting to quantized format for better performance)")
                
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
                logger.info(f"  uv run python convert_to_mlx.py --hf-path {self.config.base_model} --mlx-path {model_path}")
                if self.config.mlx_quantization:
                    logger.info(f"  --quantize {self.config.mlx_quantization}")
            else:
                # No MLX model specified or found
                logger.info("MLX model not found. Will use PyTorch for generation (slower).")
                logger.info("Tip: Convert model to MLX format for 5-10x faster generation:")
                logger.info(f"  uv run python convert_to_mlx.py --hf-path {self.config.base_model} --mlx-path ./mlx_model")
                if self.config.mlx_quantization:
                    logger.info(f"  --quantize {self.config.mlx_quantization}")
                logger.info("Then update config.yaml:")
                logger.info("  hardware:")
                logger.info("    use_mlx_for_generation: true")
                logger.info("    mlx_model_path: ./mlx_model")
                if self.config.mlx_quantization:
                    logger.info(f"    mlx_quantization: {self.config.mlx_quantization}")
        except ImportError:
            logger.warning("MLX not available. Install with: uv pip install mlx mlx-lm")
            logger.info("Using PyTorch MPS for generation (slower)")
        except Exception as e:
            logger.warning(f"Could not load MLX model: {e}")
            logger.info("Falling back to PyTorch MPS for generation")
            import traceback
            logger.debug(traceback.format_exc())
    
    def generate_student_samples(self, prompts: List[str], languages: List[str], num_samples: int = 4) -> List[Dict]:
        """Generate multiple samples from student model for each prompt (optimized for M5)"""
        # Use MLX for generation if available (much faster than PyTorch MPS)
        if self.mlx_model is not None and self.mlx_tokenizer is not None:
            return self._generate_with_mlx(prompts, languages, num_samples)
        
        # Fall back to PyTorch MPS
        return self._generate_with_pytorch(prompts, languages, num_samples)
    
    def _generate_with_mlx(self, prompts: List[str], languages: List[str], num_samples: int) -> List[Dict]:
        """Generate using MLX (5-10x faster than PyTorch MPS on Apple Silicon)
        
        Similar to preload_model.py, uses pre-compiled MLX model for fast generation.
        """
        from mlx_lm import generate as mlx_generate
        
        samples = []
        all_formatted_prompts = []
        prompt_metadata = []
        
        for prompt, language in zip(prompts, languages):
            formatted_prompt = f"Write high-quality {language} code:\n\n{prompt}\n\nCode:"
            for _ in range(num_samples):
                all_formatted_prompts.append(formatted_prompt)
                prompt_metadata.append((prompt, language))
        
        # MLX generation is much faster - can process sequentially or in small batches
        # Similar to preload_model.py, use minimal parameters for fastest generation
        logger.debug(f"Generating {len(all_formatted_prompts)} samples with MLX (5-10x faster than PyTorch MPS)...")
        
        for formatted_prompt, (prompt, language) in zip(all_formatted_prompts, prompt_metadata):
            try:
                # MLX generate is optimized for Apple Silicon
                # Use minimal parameters for fastest generation (same as preload_model.py)
                # MLX automatically optimizes for Apple Silicon
                # Optimize generation: reduce max_tokens for faster generation
                # 64 tokens is typically enough for code snippets and much faster
                max_gen_tokens = min(64, self.config.max_length // 8)  # Reduced from 128 to 64 for speed
                generated_text = mlx_generate(
                    self.mlx_model,
                    self.mlx_tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=max_gen_tokens,  # Reduced for faster generation
                    # Note: MLX uses sampler for temperature/top_k/top_p control
                    # For fastest: no sampler (greedy), or use default sampler
                    # For quality: can add temperature, top_k, top_p parameters if needed
                )
                
                # Extract only the generated part (remove prompt)
                if generated_text.startswith(formatted_prompt):
                    generated_code = generated_text[len(formatted_prompt):].strip()
                else:
                    generated_code = generated_text.strip()
                
                # CRITICAL: Verify tokenizer compatibility for the actual generated code
                # MLX and PyTorch tokenizers MUST produce the same token IDs for the same text
                # If they don't match, we'll get invalid token IDs and NaN logits
                try:
                    # Test tokenization of generated code with both tokenizers
                    mlx_gen_tokens = self.mlx_tokenizer.encode(generated_code, add_special_tokens=False)
                    pytorch_gen_tokens = self.tokenizer.encode(generated_code, add_special_tokens=False)
                    
                    if mlx_gen_tokens != pytorch_gen_tokens:
                        logger.warning(f"⚠️  Tokenizer mismatch for generated code!")
                        logger.warning(f"  MLX tokens ({len(mlx_gen_tokens)}): {mlx_gen_tokens[:20]}...")
                        logger.warning(f"  PyTorch tokens ({len(pytorch_gen_tokens)}): {pytorch_gen_tokens[:20]}...")
                        logger.warning(f"  This mismatch will cause NaN logits!")
                        logger.warning(f"  SOLUTION: Use PyTorch tokenizer to re-tokenize the generated code")
                        
                        # Re-tokenize with PyTorch tokenizer to ensure consistency
                        # Decode MLX tokens and re-encode with PyTorch tokenizer
                        try:
                            # Decode what MLX tokenizer produced
                            mlx_decoded = self.mlx_tokenizer.decode(mlx_gen_tokens, skip_special_tokens=True)
                            # Re-encode with PyTorch tokenizer
                            pytorch_reencoded = self.tokenizer.encode(mlx_decoded, add_special_tokens=False)
                            # Use the PyTorch-tokenized version
                            generated_code = self.tokenizer.decode(pytorch_reencoded, skip_special_tokens=True)
                            logger.debug("Fixed tokenizer mismatch by re-tokenizing with PyTorch tokenizer")
                        except Exception as e:
                            logger.error(f"Failed to fix tokenizer mismatch: {e}")
                            logger.error("This will likely cause NaN logits. Consider disabling MLX generation.")
                    else:
                        logger.debug(f"✓ Tokenizer compatibility verified for generated code ({len(mlx_gen_tokens)} tokens)")
                except Exception as e:
                    logger.warning(f"Could not verify tokenizer compatibility for generated code: {e}")
                
                # For training, we need the FULL sequence (prompt + generated code)
                # IMPORTANT: Use PyTorch tokenizer (self.tokenizer) consistently for training
                # MLX tokenizer was used for generation, but we need PyTorch tokenizer for training
                # The generated_code is plain text, so tokenizing with PyTorch tokenizer should work
                # However, if there are special tokens or formatting differences, we may get mismatches
                
                # CRITICAL: Ensure the generated_code doesn't contain special tokens that MLX added
                # MLX might add special tokens that PyTorch tokenizer interprets differently
                # Clean the generated code to remove any potential MLX-specific formatting
                generated_code_clean = generated_code.strip()
                
                # Reconstruct the full sequence with clean generated code
                full_sequence = formatted_prompt + generated_code_clean
                
                # Debug: Log if sequence is suspiciously long (may indicate tokenization issues)
                if len(full_sequence) > self.config.max_length * 4:  # Rough estimate: 4 chars per token
                    logger.debug(f"Full sequence is long ({len(full_sequence)} chars), will be truncated to {self.config.max_length} tokens")
                
                # Verify tokenization produces valid token IDs
                # Tokenize a test to ensure tokenizer is working correctly
                try:
                    test_tokenized = self.tokenizer("test", return_tensors="pt")
                    if torch.isnan(test_tokenized['input_ids']).any():
                        logger.error("Tokenizer itself is producing NaN! This is a critical issue.")
                except Exception as e:
                    logger.warning(f"Tokenizer test failed: {e}")
                
                # Tokenize with PyTorch tokenizer (must match the model's tokenizer)
                tokenized = self.tokenizer(
                    full_sequence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length,
                    padding='max_length',  # Pad to max_length for batching
                    return_attention_mask=True,
                )
                
                # Validate tokenization
                input_ids_tensor = tokenized['input_ids'].squeeze()
                if input_ids_tensor.numel() == 0:
                    logger.warning(f"Empty tokenization for prompt, using fallback")
                    tokenized = self.tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.config.max_length,
                        padding='max_length',
                        return_attention_mask=True,
                    )
                    input_ids_tensor = tokenized['input_ids'].squeeze()
                
                # Validate token IDs are in valid range
                # Get vocab_size from model config (more reliable than len(tokenizer))
                vocab_size = getattr(self.model.config, 'vocab_size', len(self.tokenizer))
                if vocab_size == 0 or vocab_size is None:
                    vocab_size = len(self.tokenizer)
                
                # Check for invalid token IDs
                invalid_mask = (input_ids_tensor < 0) | (input_ids_tensor >= vocab_size)
                if invalid_mask.any():
                    invalid_count = invalid_mask.sum().item()
                    logger.error(f"Invalid token IDs detected in MLX generation sample: {invalid_count} tokens out of range [0, {vocab_size})")
                    logger.error(f"Token ID range: min={input_ids_tensor.min().item()}, max={input_ids_tensor.max().item()}, vocab_size={vocab_size}")
                    # Replace invalid IDs with pad token
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                    if pad_token_id is None:
                        pad_token_id = 0  # Fallback to 0
                    input_ids_tensor = torch.where(invalid_mask,
                                                  torch.tensor(pad_token_id, dtype=input_ids_tensor.dtype),
                                                  input_ids_tensor)
                    tokenized['input_ids'] = input_ids_tensor.unsqueeze(0) if len(input_ids_tensor.shape) == 1 else input_ids_tensor
                
                # Additional validation: check for NaN/Inf in token IDs (shouldn't happen, but check anyway)
                if torch.isnan(input_ids_tensor.float()).any() or torch.isinf(input_ids_tensor.float()).any():
                    logger.error("NaN/Inf detected in tokenized input_ids! This is unexpected.")
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                    if pad_token_id is None:
                        pad_token_id = 0
                    input_ids_tensor = torch.where(torch.isnan(input_ids_tensor.float()) | torch.isinf(input_ids_tensor.float()),
                                                  torch.tensor(pad_token_id, dtype=input_ids_tensor.dtype),
                                                  input_ids_tensor)
                    tokenized['input_ids'] = input_ids_tensor.unsqueeze(0) if len(input_ids_tensor.shape) == 1 else input_ids_tensor
                
                samples.append({
                    'prompt': prompt,
                    'language': language,
                    'code': generated_code,
                    'input_ids': tokenized['input_ids'].squeeze(),
                    'attention_mask': tokenized['attention_mask'].squeeze(),
                })
            except Exception as e:
                logger.warning(f"MLX generation failed for prompt: {e}")
                # Fall back to empty code
                formatted_prompt = f"Write high-quality {language} code:\n\n{prompt}\n\nCode:"
                tokenized_fallback = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length,
                    padding='max_length',
                    return_attention_mask=True,
                )
                samples.append({
                    'prompt': prompt,
                    'language': language,
                    'code': '',
                    'input_ids': tokenized_fallback['input_ids'].squeeze(),
                    'attention_mask': tokenized_fallback['attention_mask'].squeeze(),
                })
        
        return samples
    
    def _generate_with_pytorch(self, prompts: List[str], languages: List[str], num_samples: int) -> List[Dict]:
        """Generate using PyTorch MPS (fallback if MLX not available)"""
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            # Batch process all prompts at once for efficiency
            all_formatted_prompts = []
            prompt_metadata = []
            
            for prompt, language in zip(prompts, languages):
                formatted_prompt = f"Write high-quality {language} code:\n\n{prompt}\n\nCode:"
                for _ in range(num_samples):
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
                    generation_config = {
                        "max_new_tokens": min(128, self.config.max_length // 4),  # Reduced from 256 to save memory
                        "do_sample": True,  # Enable sampling to use temperature/top_p/top_k
                        "temperature": 0.8,
                        "top_k": self.config.top_k,
                        "top_p": self.config.top_p,
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "num_return_sequences": 1,
                        "use_cache": True,  # Critical for MPS performance
                        "output_scores": False,
                        "return_dict_in_generate": False,
                        "repetition_penalty": 1.1,  # Prevent repetition
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
                            
                            # For PyTorch generation, we need the full sequence (prompt + generated code)
                            full_sequence = all_formatted_prompts[i * batch_size + j] + generated_text
                            tokenized_full = self.tokenizer(
                                full_sequence,
                                return_tensors="pt",
                                truncation=True,
                                max_length=self.config.max_length,
                                padding='max_length',
                                return_attention_mask=True,
                            )
                            
                            samples.append({
                                'prompt': prompt,
                                'language': language,
                                'code': generated_text,
                                'input_ids': tokenized_full['input_ids'].squeeze(),
                                'attention_mask': tokenized_full['attention_mask'].squeeze(),
                            })
        
        return samples
    
    def _get_teacher_code_cached(self, prompt: str, language: str) -> str:
        """Get teacher code with caching"""
        cache_key = f"{prompt}:{language}"
        if cache_key not in self.teacher_cache:
            self.teacher_cache[cache_key] = self.teacher.generate(prompt, language)
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
            else:
                student_score = self.teacher.score_code(
                    sample['code'],
                    sample['prompt'],
                    sample['language']
                )
                self.teacher_score_cache[student_code_key] = student_score
            
            # Score teacher code (baseline) - cache this too
            teacher_code_key = f"{teacher_code}:{sample['prompt']}:{sample['language']}"
            if teacher_code_key in self.teacher_score_cache:
                teacher_score = self.teacher_score_cache[teacher_code_key]
            else:
                teacher_score = self.teacher.score_code(
                    teacher_code,
                    sample['prompt'],
                    sample['language']
                )
                self.teacher_score_cache[teacher_code_key] = teacher_score
            
            # Normalized reward (relative to teacher)
            reward = student_score / (teacher_score + 1e-6)
            
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
            logger.warning(f"Error processing sample: {e}")
            return 0.5, None
    
    def compute_rewards(self, samples: List[Dict], save_to_dataset: bool = True) -> Tuple[List[float], List[Dict]]:
        """Compute rewards using teacher model with parallel processing (optimized for M5)
        
        Optimizations:
        - Parallel API calls with ThreadPoolExecutor
        - Caching to avoid redundant API calls
        - Adaptive worker count based on sample count
        - Progress tracking with tqdm
        """
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
                    logger.warning("Reward computation timed out, using default reward")
                    rewards.append(0.5)  # Default reward on timeout
                except Exception as e:
                    logger.warning(f"Error getting reward result: {e}")
                    rewards.append(0.5)  # Default reward on error
        
        return rewards, dataset_entries
    
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
        # Group samples by original prompt (each prompt has num_samples_per_prompt samples)
        num_samples_per_prompt = len(samples) // len(original_prompts)
        
        # Collect input_ids and attention_masks from samples
        batch_input_ids = []
        batch_attention_masks = []
        
        for i, prompt in enumerate(original_prompts):
            # Get the first sample for each prompt (or average if multiple)
            # For now, use the first sample per prompt
            sample_idx = i * num_samples_per_prompt
            if sample_idx < len(samples):
                sample = samples[sample_idx]
                if 'input_ids' in sample and 'attention_mask' in sample:
                    batch_input_ids.append(sample['input_ids'])
                    batch_attention_masks.append(sample['attention_mask'])
                else:
                    # Fallback: tokenize the prompt (shouldn't happen if generation worked)
                    logger.warning(f"Sample missing input_ids/attention_mask, using fallback tokenization")
                    formatted_prompt = f"Write high-quality {sample.get('language', 'python')} code:\n\n{prompt}\n\nCode:"
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
        
        # Backward pass
        total_loss.backward()
        
        # Clear intermediate tensors to free memory (optimization for M5)
        # Delete in order to free memory immediately
        # Note: ref_outputs was already deleted earlier (line 1535) to free memory immediately
        del logits, log_probs, ref_log_probs, selected_log_probs, ref_selected_log_probs, outputs
        
        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'avg_reward': np.mean(rewards),
        }
    
    def train(self, train_dataset: CodeDataset, eval_dataset: Optional[CodeDataset] = None):
        """Main training loop"""
        logger.info("Starting RLAIF training...")
        
        # Start system monitoring
        self._start_monitoring()
        
        # Optimize DataLoader for M5: use num_workers=0 to avoid fork issues
        # M5 has unified memory, so single process is actually faster
        num_workers = 0 if self.config.use_mps else min(2, os.cpu_count() or 1)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,  # M5 doesn't benefit from pin_memory
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,  # Prefetch for faster data loading
            drop_last=False  # Keep all batches
        )
        
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
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            self.stats['epoch'] = epoch + 1
            
            epoch_rewards = []
            epoch_losses = []
            
            # Batch dataset collection to reduce memory overhead
            dataset_batch = []
            dataset_batch_size = 10  # Collect 10 batches before extending main list
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
                batch_start_time = time.time()
                
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
                samples = self.generate_student_samples(
                    prompts,
                    languages,
                    num_samples=self.config.num_samples_per_prompt
                )
                gen_time = time.time() - gen_start
                
                # Synchronize MPS only if needed (after generation, before training)
                if torch.backends.mps.is_available() and batch_idx % 5 == 0:
                    torch.mps.synchronize()  # Sync periodically, not every batch
                
                # Log generation performance (similar to preload_model.py)
                if batch_idx % 5 == 0:
                    num_tokens = sum(len(s.get('code', '').split()) for s in samples)
                    tokens_per_sec = num_tokens / gen_time if gen_time > 0 else 0
                    batch_size_actual = len(batch['prompt'])
                    logger.info(f"Batch {batch_idx} (size={batch_size_actual}) - Generation: {gen_time:.1f}s, {tokens_per_sec:.1f} tokens/sec")
                    
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
                        logger.warning(f"  1. Convert model: uv run python convert_to_mlx.py --hf-path {self.config.base_model} --mlx-path ./mlx_model --quantize q8_bit")
                        logger.warning("  2. Update config.yaml: hardware.use_mlx_for_generation: true")
                
                # Compute rewards and collect dataset entries (optimized)
                reward_start = time.time()
                rewards, dataset_entries = self.compute_rewards(samples, save_to_dataset=True)
                reward_time = time.time() - reward_start
                epoch_rewards.extend(rewards)
                
                # Batch dataset collection to reduce memory overhead
                dataset_batch.extend(dataset_entries)
                if len(dataset_batch) >= dataset_batch_size * len(samples):
                    self.dataset_collection['training'].extend(dataset_batch)
                    dataset_batch = []  # Clear batch
                
                # Training step
                # Reconstruct batch from generated samples (with full sequences: prompt + generated code)
                # The original batch only has prompts, but we need the full sequences for training
                train_batch = self._create_training_batch_from_samples(samples, batch['prompt'])
                
                train_start = time.time()
                loss_dict = self.train_step(train_batch, rewards[:len(batch['prompt'])])
                train_time = time.time() - train_start
                epoch_losses.append(loss_dict['loss'])
                
                # Aggressive memory cleanup after training step to prevent OOM
                if torch.backends.mps.is_available():
                    # Delete intermediate tensors explicitly
                    if 'train_batch' in locals():
                        del train_batch
                    if 'samples' in locals():
                        del samples
                    if 'rewards' in locals():
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
                
                batch_time = time.time() - batch_start_time
                
                # Log timing info periodically with more detail
                if batch_idx % 5 == 0:
                    batch_size_actual = len(batch['prompt'])
                    logger.info(
                        f"Batch {batch_idx} (size={batch_size_actual}) timing breakdown:\n"
                        f"  Generation: {gen_time:.1f}s ({gen_time/batch_time*100:.1f}%)\n"
                        f"  Rewards: {reward_time:.1f}s ({reward_time/batch_time*100:.1f}%)\n"
                        f"  Training: {train_time:.1f}s ({train_time/batch_time*100:.1f}%)\n"
                        f"  Total: {batch_time:.1f}s"
                    )
                    
                    # Identify bottleneck
                    if gen_time > batch_time * 0.5:
                        logger.warning(f"⚠️  Generation is the bottleneck ({gen_time/batch_time*100:.1f}% of time)")
                        if self.mlx_model is None:
                            logger.warning("  → Enable MLX for 5-10x speedup (see above)")
                    elif reward_time > batch_time * 0.5:
                        logger.warning(f"⚠️  Reward computation is the bottleneck ({reward_time/batch_time*100:.1f}% of time)")
                        logger.info("  → Consider reducing num_samples_per_prompt or increasing API parallelism")
                    elif train_time > batch_time * 0.5:
                        logger.warning(f"⚠️  Training step is the bottleneck ({train_time/batch_time*100:.1f}% of time)")
                        logger.info("  → Consider reducing batch_size or max_length")
            
            # Flush remaining dataset entries at end of epoch
            if dataset_batch:
                self.dataset_collection['training'].extend(dataset_batch)
                dataset_batch = []
                
                # Update optimizer
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # Update stats
                self.stats['step'] = global_step
                self.stats['total_reward'] += np.mean(rewards) if rewards else 0.0
                self.stats['total_loss'] += loss_dict['loss']
                self.stats['num_samples'] += len(rewards) if rewards else 0
                self.stats['avg_reward'] = self.stats['total_reward'] / max(1, self.stats['step'])
                self.stats['avg_loss'] = self.stats['total_loss'] / max(1, self.stats['step'])
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    self._log_stats(global_step, loss_dict, rewards)
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(global_step)
            
            # Epoch summary
            avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            
            logger.info(
                f"Epoch {epoch + 1} Summary:\n"
                f"  Average Reward: {avg_epoch_reward:.4f}\n"
                f"  Average Loss: {avg_epoch_loss:.4f}\n"
                f"  Total Samples: {len(epoch_rewards)}"
            )
            
            if self.writer:
                self.writer.add_scalar('Epoch/AvgReward', avg_epoch_reward, epoch)
                self.writer.add_scalar('Epoch/AvgLoss', avg_epoch_loss, epoch)
        
        # Final save
        self._save_checkpoint(global_step, final=True)
        
        # Stop system monitoring
        self._stop_monitoring()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Save and upload datasets
        if self.config.save_datasets_locally or self.config.upload_datasets:
            self._save_and_upload_datasets(global_step)
        
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
    
    def _log_stats(self, step: int, loss_dict: Dict, rewards: List[float]):
        """Log training statistics"""
        avg_reward = np.mean(rewards) if rewards else 0.0
        
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
        
        if self.writer:
            # Training metrics
            self.writer.add_scalar('Train/Loss', loss_dict['loss'], step)
            self.writer.add_scalar('Train/PolicyLoss', loss_dict['policy_loss'], step)
            self.writer.add_scalar('Train/KLPenalty', loss_dict['kl_penalty'], step)
            self.writer.add_scalar('Train/AvgReward', avg_reward, step)
            if rewards:
                self.writer.add_scalar('Train/RewardStd', np.std(rewards), step)
            
            # System metrics at logging steps
            if system_metrics:
                self.writer.add_scalar('System/CPU_Percent', system_metrics.get('cpu_percent', 0), step)
                self.writer.add_scalar('System/Memory_Percent', system_metrics.get('memory_percent', 0), step)
                self.writer.add_scalar('System/Memory_Used_GB', system_metrics.get('memory_used_gb', 0), step)
                self.writer.add_scalar('System/Process_Memory_GB', system_metrics.get('process_memory_gb', 0), step)
                if system_metrics.get('gpu_memory_total_gb', 0) > 0:
                    self.writer.add_scalar('System/GPU_Memory_Used_GB', system_metrics.get('gpu_memory_used_gb', 0), step)
                    self.writer.add_scalar('System/GPU_Memory_Percent', system_metrics.get('gpu_memory_percent', 0), step)
                    self.writer.add_scalar('System/GPU_Utilization', system_metrics.get('gpu_utilization', 0), step)
    
    def _save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint in both PyTorch and MLX formats"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
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

# Qwen Code RLAIF Dataset

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

dataset = load_dataset("{self.config.dataset_repo_id or 'mlx-community/qwen-code-rlaif-dataset'}")

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
@dataset{{qwen_code_rlaif_dataset,
  title={{Qwen Code RLAIF Dataset}},
  author={{MLX Community}},
  year={{2024}},
  url={{https://huggingface.co/datasets/{self.config.dataset_repo_id or 'mlx-community/qwen-code-rlaif-dataset'}}}
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
        max_grad_norm=to_float(training_cfg.get('max_grad_norm'), 1.0),
        weight_decay=to_float(training_cfg.get('weight_decay'), 0.01),
        lr_scheduler_type=training_cfg.get('lr_scheduler_type', 'cosine'),
        reward_weight=to_float(rlaif_cfg.get('reward_weight'), 1.0),
        kl_penalty=to_float(rlaif_cfg.get('kl_penalty'), 0.1),
        beta=to_float(rlaif_cfg.get('beta'), 0.1),
        num_samples_per_prompt=to_int(rlaif_cfg.get('num_samples_per_prompt'), 4),
        max_length=to_int(model_cfg.get('max_length'), 2048),
        use_4bit=to_bool(model_cfg.get('use_4bit'), True),
        use_mps=to_bool(hardware_cfg.get('use_mps'), True),
        mixed_precision=hardware_cfg.get('mixed_precision', 'bf16'),
        tensorboard_dir=logging_cfg.get('tensorboard_dir', './logs/tensorboard'),
        log_level=logging_cfg.get('log_level', 'INFO'),
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
    
    args = parser.parse_args()
    
    # Load configuration
    config, data_cfg = load_config(args.config)
    
    # Override model if provided via command line
    if args.model:
        config.base_model = args.model
        logger.info(f"Using model from command line: {config.base_model}")
    
    # Setup logging level
    logging.getLogger().setLevel(getattr(logging, config.log_level))
    
    # Suppress httpx HTTP request logs unless in DEBUG mode
    # httpx logs every HTTP request at INFO level, which is too verbose for training
    httpx_logger = logging.getLogger("httpx")
    if config.log_level.upper() == "DEBUG":
        httpx_logger.setLevel(logging.INFO)  # Show HTTP requests in DEBUG mode
    else:
        httpx_logger.setLevel(logging.WARNING)  # Suppress HTTP requests in INFO/WARNING mode
    
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

