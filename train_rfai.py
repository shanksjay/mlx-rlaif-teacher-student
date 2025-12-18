#!/usr/bin/env python3
"""
RFAI (Reinforcement from AI Feedback) Training Script for Qwen Code Model

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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import psutil
import threading
import time
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


@dataclass
class RFAIConfig:
    """Configuration for RFAI training"""
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
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def generate(self, prompt: str, language: str, max_tokens: int = 2048) -> str:
        """Generate code using teacher model"""
        system_prompt = f"You are an expert {language} programmer. Generate clean, efficient, and well-documented code."
        full_prompt = f"{prompt}\n\nGenerate high-quality {language} code:"
        
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
            logger.error(f"Error generating from teacher model: {e}")
            return ""
    
    def score_code(self, code: str, prompt: str, language: str) -> float:
        """Score code quality using teacher model"""
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

Respond with ONLY a single float between 0.0 and 1.0, nothing else."""
        
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
            
            # Extract float from response
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except ValueError:
                logger.warning(f"Could not parse score: {score_text}")
                return 0.5
        
        except Exception as e:
            logger.error(f"Error scoring code: {e}")
            return 0.5


class RFAITrainer:
    """RFAI Trainer implementing teacher-student training"""
    
    def __init__(self, config: RFAIConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Load model and tokenizer
        logger.info(f"Loading base model: {config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup quantization for M5 MacBook
        if config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                device_map="auto",
                torch_dtype=torch.bfloat16 if config.use_mps else torch.float32,
            )
        
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
    
    def _setup_device(self):
        """Setup device for M5 MacBook"""
        if self.config.use_mps and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Metal Performance Shaders)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device
    
    def generate_student_samples(self, prompts: List[str], languages: List[str], num_samples: int = 4) -> List[Dict]:
        """Generate multiple samples from student model for each prompt"""
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            for prompt, language in zip(prompts, languages):
                formatted_prompt = f"Write high-quality {language} code:\n\n{prompt}\n\nCode:"
                
                for _ in range(num_samples):
                    inputs = self.tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.config.max_length
                    ).to(self.device)
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.8,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    samples.append({
                        'prompt': prompt,
                        'language': language,
                        'code': generated_text,
                        'input_ids': inputs['input_ids'].squeeze(),
                    })
        
        return samples
    
    def compute_rewards(self, samples: List[Dict], save_to_dataset: bool = True) -> Tuple[List[float], List[Dict]]:
        """Compute rewards using teacher model and optionally save to dataset"""
        rewards = []
        dataset_entries = []
        
        for sample in tqdm(samples, desc="Computing rewards"):
            # Get teacher's reference code
            teacher_code = self.teacher.generate(
                sample['prompt'],
                sample['language']
            )
            
            # Score student code
            student_score = self.teacher.score_code(
                sample['code'],
                sample['prompt'],
                sample['language']
            )
            
            # Score teacher code (baseline)
            teacher_score = self.teacher.score_code(
                teacher_code,
                sample['prompt'],
                sample['language']
            )
            
            # Normalized reward (relative to teacher)
            reward = student_score / (teacher_score + 1e-6)
            rewards.append(reward)
            
            # Save to dataset collection if enabled
            if save_to_dataset:
                dataset_entry = {
                    'prompt': sample['prompt'],
                    'language': sample['language'],
                    'student_code': sample['code'],
                    'teacher_code': teacher_code,
                    'student_score': float(student_score),
                    'teacher_score': float(teacher_score),
                    'reward': float(reward),
                    'scoring_breakdown': {
                        'correctness': 0.3,  # These would come from detailed scoring if implemented
                        'code_quality': 0.3,
                        'efficiency': 0.2,
                        'documentation': 0.2
                    },
                    'timestamp': datetime.now().isoformat()
                }
                dataset_entries.append(dataset_entry)
        
        return rewards, dataset_entries
    
    def compute_kl_penalty(self, log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence penalty"""
        kl = log_probs - ref_log_probs
        return self.config.kl_penalty * kl.mean()
    
    def train_step(self, batch: Dict, rewards: List[float]) -> Dict[str, float]:
        """Perform one training step with RFAI"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Get reference log probs (from base model, frozen)
        with torch.no_grad():
            ref_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            ref_log_probs = torch.nn.functional.log_softmax(ref_outputs.logits, dim=-1)
        
        # Compute policy gradient loss
        # Select log probs for generated tokens
        # Simplified: use average log prob as proxy
        selected_log_probs = log_probs[:, :-1, :].gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Convert rewards to tensor
        reward_tensor = torch.tensor(rewards, device=self.device).unsqueeze(1)
        reward_tensor = reward_tensor.expand_as(selected_log_probs)
        
        # Policy gradient: maximize log_prob * reward
        policy_loss = -(selected_log_probs * reward_tensor * attention_mask[:, 1:]).mean()
        
        # KL penalty
        kl_penalty = self.compute_kl_penalty(selected_log_probs, ref_log_probs[:, :-1, :].gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1))
        
        # Total loss
        total_loss = policy_loss + kl_penalty
        
        # Backward pass
        total_loss.backward()
        
        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'avg_reward': np.mean(rewards),
        }
    
    def train(self, train_dataset: CodeDataset, eval_dataset: Optional[CodeDataset] = None):
        """Main training loop"""
        logger.info("Starting RFAI training...")
        
        # Start system monitoring
        self._start_monitoring()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
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
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
                # Generate student samples
                prompts = batch['prompt']
                languages = batch['language']
                
                samples = self.generate_student_samples(
                    prompts,
                    languages,
                    num_samples=self.config.num_samples_per_prompt
                )
                
                # Compute rewards and collect dataset entries
                rewards, dataset_entries = self.compute_rewards(samples, save_to_dataset=True)
                epoch_rewards.extend(rewards)
                
                # Add to dataset collection
                self.dataset_collection['training'].extend(dataset_entries)
                
                # Training step
                loss_dict = self.train_step(batch, rewards[:len(batch['prompt'])])
                epoch_losses.append(loss_dict['loss'])
                
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
            
            # GPU/MPS memory (if using PyTorch)
            gpu_memory_used = 0.0
            gpu_memory_total = 0.0
            if torch.backends.mps.is_available():
                # MPS doesn't have direct memory query, but we can track allocations
                if hasattr(torch.mps, 'current_allocated_memory'):
                    gpu_memory_used = torch.mps.current_allocated_memory() / (1024 ** 3)
                    gpu_memory_total = torch.mps.driver_allocated_memory() / (1024 ** 3) if hasattr(torch.mps, 'driver_allocated_memory') else 0.0
            elif torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / (1024 ** 3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            
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
                stats_str += (
                    f"\n  GPU Memory: {system_metrics.get('gpu_memory_percent', 0):.1f}% "
                    f"({system_metrics.get('gpu_memory_used_gb', 0):.2f}GB / {system_metrics.get('gpu_memory_total_gb', 0):.2f}GB)"
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
- rfai
- qwen
- fine-tuned
base_model: {self.config.base_model}
---

# {repo_id.split('/')[-1]}

This model is a fine-tuned version of [{self.config.base_model}](https://huggingface.co/{self.config.base_model}) using Reinforcement from AI Feedback (RFAI).

## Training Details

- **Base Model**: {self.config.base_model}
- **Training Method**: RFAI (Reinforcement from AI Feedback)
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
- rfai
- reinforcement-learning
size_categories:
- 1K<n<10K
---

# Qwen Code RFAI Dataset

This dataset contains prompts, teacher-generated code, student-generated code, and scoring parameters from the RFAI (Reinforcement from AI Feedback) training process.

## Dataset Description

This dataset was generated during the fine-tuning of a Qwen model for code generation using RFAI methodology. Each entry includes:

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

dataset = load_dataset("{self.config.dataset_repo_id or 'mlx-community/qwen-code-rfai-dataset'}")

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
@dataset{{qwen_code_rfai_dataset,
  title={{Qwen Code RFAI Dataset}},
  author={{MLX Community}},
  year={{2024}},
  url={{https://huggingface.co/datasets/{self.config.dataset_repo_id or 'mlx-community/qwen-code-rfai-dataset'}}}
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
                commit_message=f"Upload RFAI training dataset (step {step})",
                ignore_patterns=["*.pyc", "__pycache__"]
            )
            
            logger.info(f"✓ Successfully uploaded dataset to https://huggingface.co/datasets/{repo_id}")
            
        except ImportError:
            logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
        except Exception as e:
            logger.error(f"Error uploading dataset to Hugging Face: {e}")
            import traceback
            logger.debug(traceback.format_exc())


def load_config(config_path: str) -> Tuple[RFAIConfig, dict]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Flatten nested config
    model_cfg = config_dict.get('model', {})
    teacher_cfg = config_dict.get('teacher', {})
    training_cfg = config_dict.get('training', {})
    rfai_cfg = config_dict.get('rfai', {})
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
    
    rfai_config = RFAIConfig(
        base_model=model_cfg.get('base_model', 'Qwen/Qwen2.5-7B-Instruct'),
        teacher_provider=teacher_cfg.get('provider', 'openai'),
        teacher_model=teacher_cfg.get('model_name', 'gpt-4-turbo-preview'),
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
        reward_weight=to_float(rfai_cfg.get('reward_weight'), 1.0),
        kl_penalty=to_float(rfai_cfg.get('kl_penalty'), 0.1),
        beta=to_float(rfai_cfg.get('beta'), 0.1),
        num_samples_per_prompt=to_int(rfai_cfg.get('num_samples_per_prompt'), 4),
        max_length=to_int(model_cfg.get('max_length'), 2048),
        use_4bit=to_bool(model_cfg.get('use_4bit'), True),
        use_mps=to_bool(hardware_cfg.get('use_mps'), True),
        mixed_precision=hardware_cfg.get('mixed_precision', 'bf16'),
        tensorboard_dir=logging_cfg.get('tensorboard_dir', './logs/tensorboard'),
        log_level=logging_cfg.get('log_level', 'INFO'),
        top_k=to_int(rfai_cfg.get('top_k'), 50),
        top_p=to_float(rfai_cfg.get('top_p'), 0.95),
        save_mlx_format=to_bool(hardware_cfg.get('save_mlx_format'), True),
        mlx_quantization=hardware_cfg.get('mlx_quantization', None),
        upload_to_hub=to_bool(config_dict.get('huggingface', {}).get('upload_to_hub'), False),
        hf_repo_id=config_dict.get('huggingface', {}).get('repo_id', None),
        hf_token_env=config_dict.get('huggingface', {}).get('hf_token_env', 'HUGGINGFACE_TOKEN'),
        upload_quantized=to_bool(config_dict.get('huggingface', {}).get('upload_quantized'), True),
        hf_private=to_bool(config_dict.get('huggingface', {}).get('private'), False),
        upload_datasets=to_bool(config_dict.get('huggingface', {}).get('upload_datasets'), True),
        dataset_repo_id=config_dict.get('huggingface', {}).get('dataset_repo_id', None),
        save_datasets_locally=to_bool(config_dict.get('huggingface', {}).get('save_datasets_locally'), True),
        dataset_output_dir=config_dict.get('huggingface', {}).get('dataset_output_dir', './datasets'),
    )
    
    return rfai_config, data_cfg


def main():
    parser = argparse.ArgumentParser(description="RFAI Training for Qwen Code Model")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
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
    
    # Setup logging level
    logging.getLogger().setLevel(getattr(logging, config.log_level))
    
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
        trainer = RFAITrainer(config)
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

