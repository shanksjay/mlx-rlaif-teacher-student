# Code Fine-tuning with RLAIF (Reinforcement Learning from AI Feedback)

This project implements a teacher-student training scheme to fine-tune any generative AI model for generating high-quality C++, Python, and Rust code using Reinforcement Learning from AI Feedback (RLAIF).

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [High-Level Flow](#high-level-flow)
- [Architecture Details](#architecture-details)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Performance Optimizations](#performance-optimizations)
- [MLX Optimization (5-10x Faster Generation)](#mlx-optimization-5-10x-faster-generation)
- [Model Validation](#model-validation)
- [Dataset Collection and Upload](#dataset-collection-and-upload)
- [Hugging Face Model Upload](#hugging-face-model-upload)
- [TensorBoard Metrics](#tensorboard-metrics)
- [Profiling and Debugging](#profiling-and-debugging)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [References](#references)

## Overview

The training system uses a two-model architecture:
- **Teacher Model**: OpenAI Codex (GPT-4) or Claude - provides high-quality reference code and scoring
- **Student Model**: Smaller GenAI model (default: Qwen2.5-Coder-3B-Instruct) - being fine-tuned to match teacher quality. Can be customized via `--model` argument.

## Quick Start

### Prerequisites Check

1. **Python 3.9+**: Check with `python3 --version`
2. **API Key**: Set either:
   ```bash
   export OPENAI_API_KEY="sk-..."
   # OR
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

### Installation (5 minutes)

```bash
# Install dependencies
uv pip install -r requirements.txt

# (Optional) Preload model for faster startup
uv run python scripts/utils/preload_model.py --model Qwen/Qwen2.5-Coder-3B-Instruct

# Generate sample data
uv run python scripts/utils/data_utils.py
```

### Run Training

#### Option 1: Using the convenience script
```bash
./run_training.sh
```

#### Option 2: Manual execution
```bash
# Start TensorBoard in background (optional)
# Suppress pkg_resources deprecation warning
PYTHONWARNINGS=ignore::UserWarning uv run tensorboard --logdir ./logs/tensorboard &

# Run training
uv run python scripts/training/train_rlaif.py --config config.yaml
```

### Monitor Training

1. **Console Output**: Watch for printed statistics:
   ```
   Step 50:
     Loss: 2.3456
     Policy Loss: 1.2345
     KL Penalty: 0.1111
     Avg Reward: 0.7890
     GPU Utilization: 75.3%
   ```

2. **TensorBoard**: Run `PYTHONWARNINGS=ignore::UserWarning uv run tensorboard --logdir ./logs/tensorboard` and open http://localhost:6006
   - To suppress the pkg_resources deprecation warning, use the PYTHONWARNINGS environment variable
   - View loss curves
   - Track reward progression
   - Monitor KL divergence
   - Monitor GPU utilization and system metrics

### Expected Output

```
2025-12-XX XX:XX:XX - INFO - Loading base model: Qwen/Qwen2.5-Coder-3B-Instruct
2025-12-XX XX:XX:XX - INFO - Using MPS (Metal Performance Shaders)
2025-12-XX XX:XX:XX - INFO - Initializing teacher model: anthropic/claude-3-5-haiku-20241022
2025-12-XX XX:XX:XX - INFO - Starting RLAIF training...
2025-12-XX XX:XX:XX - INFO - Epoch 1/3
...
```

### First Run Tips

1. **Start Small**: Use `num_samples_per_prompt: 2` in config for faster testing
2. **Monitor API Costs**: Teacher API calls can be expensive
3. **Check Memory**: M5 MacBook should handle batch_size=2 with 4-bit quantization
4. **Save Checkpoints**: Checkpoints are saved every `save_steps` (default: 500)
5. **Enable MLX**: For 5-10x faster generation, convert model to MLX format (see MLX Optimization section)

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLAIF Training Pipeline                      │
└─────────────────────────────────────────────────────────────────┘

1. Data Preparation
   ├── Load code prompts (Python, C++, Rust)
   ├── Format prompts with language context
   └── Create training/evaluation splits

2. Student Generation
   ├── For each prompt, generate N samples from Qwen model
   ├── Use temperature sampling for diversity
   └── Collect generated code samples

3. Teacher Evaluation
   ├── Teacher generates reference code for each prompt
   ├── Teacher scores student code on multiple dimensions:
   │   ├── Correctness (30%)
   │   ├── Code Quality (30%)
   │   ├── Efficiency (20%)
   │   └── Documentation (20%)
   └── Compute normalized rewards (student_score / teacher_score)

4. Policy Gradient Training
   ├── Compute log probabilities of generated tokens
   ├── Apply reward-weighted policy gradient:
   │   Loss = -log_prob * reward
   ├── Add KL divergence penalty to prevent drift:
   │   Loss += kl_penalty * KL(student || base_model)
   └── Backpropagate and update weights

5. Monitoring & Logging
   ├── Track metrics: reward, loss, KL divergence
   ├── Log to TensorBoard for visualization
   ├── Save checkpoints periodically
   └── Print critical stats to console

6. Evaluation
   ├── Generate code on evaluation set
   ├── Compute average rewards
   └── Compare against baseline
```

## Architecture Details

### Reward Computation
The reward is computed as:
```
reward = student_score / (teacher_score + ε)
```
This normalizes rewards relative to teacher performance, ensuring the student learns to match or exceed teacher quality.

### Loss Function
The training loss combines:
1. **Policy Gradient Loss**: `-log_prob * reward` (maximize high-reward outputs)
2. **KL Penalty**: `β * KL(P_student || P_base)` (prevent catastrophic forgetting)

### Training Loop
```
For each epoch:
  For each batch:
    1. Generate N student samples per prompt
    2. Get teacher reference code
    3. Score all samples using teacher
    4. Compute rewards
    5. Compute policy gradient loss
    6. Add KL penalty
    7. Backpropagate and update
    8. Log metrics
```

## Setup

### Prerequisites
- MacBook M5 (or compatible Apple Silicon)
- Python 3.9+
- CUDA/MPS support (for GPU acceleration)

### Installation

1. Clone the repository:
```bash
cd /Users/shanks108/Development/train_coding_asst
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

3. Set up API keys:
```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# OR for Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

4. Prepare data:
```bash
# Create sample dataset (or use your own)
uv run python scripts/utils/data_utils.py
```

## Configuration

Edit `config.yaml` to customize training:

- **Model**: Choose base Qwen model, quantization settings
- **Teacher**: Select OpenAI or Anthropic, model variant
- **Training**: Batch size, learning rate, epochs, etc.
- **RLAIF**: Reward weights, KL penalty, sampling parameters
- **Hardware**: MPS settings for M5 MacBook, MLX configuration
- **Logging**: TensorBoard + **offline JSONL summaries** (`logging.save_json_summaries`, `logging.json_summaries_dir`)

### Checkpoints (epoch/batch tagging + baseline gain)

By default, checkpoints are saved using `training.save_steps`, but if you use **large gradient accumulation**, `global_step` may advance slowly. This repo adds **epoch/batch-based checkpointing** so you always get more frequent checkpoints.

Config keys:

- `training.save_every_epochs`: save at end of each epoch (recommended: `1`)
- `training.save_every_batches`: save every N batches (set `0` to disable)
- `training.save_total_limit`: keep only the most recent N checkpoints (best-effort)
- `logging.baseline_eval_batches`: compute a pre-training baseline reward on the first N batches (set `0` to disable)

Checkpoint folder names include context to avoid overwriting `checkpoint-0`, e.g.:

- `checkpoint-e2-end-gs1`
- `checkpoint-e2-b10-gs1`
- `checkpoint-gs500-e3-b12`
- `checkpoint-final-gs7`

Each checkpoint directory includes:

- `training_stats.json`: rolling trainer stats
- `checkpoint_summary.json`: epoch/batch/global_step + **baseline_reward** + **reward_gain_from_baseline** (when available)

## Usage

The training script is generic and works with any compatible model. By default, it uses `Qwen/Qwen2.5-Coder-3B-Instruct` (optimized for M5 MacBook), but you can specify any model via the `--model` argument.

### Basic Training

```bash
uv run python scripts/training/train_rlaif.py --config config.yaml
```

### Custom Model

You can specify a different model using the `--model` argument. This makes the pipeline generic and works with any compatible model:

```bash
# Use a different Qwen model
uv run python scripts/training/train_rlaif.py --config config.yaml --model Qwen/Qwen2.5-7B-Instruct

# Use a local model path
uv run python scripts/training/train_rlaif.py --config config.yaml --model ./my_local_model

# Use any HuggingFace model
uv run python scripts/training/train_rlaif.py --config config.yaml --model microsoft/phi-2
```

**Recommended Models for M5 MacBook (32GB):**
- `Qwen/Qwen2.5-Coder-3B-Instruct` (default) - Fastest, best for development
- `Qwen/Qwen2.5-7B-Instruct` - Better quality, slower
- `Qwen/Qwen2.5-Coder-7B-Instruct` - Best code quality, requires more memory

### Custom Data Files

```bash
uv run python scripts/training/train_rlaif.py \
    --config config.yaml \
    --train_file ./data/my_train.jsonl \
    --eval_file ./data/my_eval.jsonl
```

### Combined Options

```bash
uv run python scripts/training/train_rlaif.py \
    --config config.yaml \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_file ./data/my_train.jsonl \
    --eval_file ./data/my_eval.jsonl
```

### Monitor Training

In a separate terminal, start TensorBoard:
```bash
# Suppress pkg_resources deprecation warning
PYTHONWARNINGS=ignore::UserWarning uv run tensorboard --logdir ./logs/tensorboard
```

Then open http://localhost:6006 in your browser.

For offline analysis (batch-by-batch and epoch-by-epoch), enable JSON summaries in `config.yaml` and read:

- `./logs/json_summaries/batches.jsonl`
- `./logs/json_summaries/epochs.jsonl`

#### Training Summary Output

At the end of training, you'll see a comprehensive summary including:

```
📊 Generation Performance:
  Average: 31.43 tokens/sec
  P99:     58.37 tokens/sec
  Samples:  350
  Total Tokens Generated: 76,505

🔄 Backpropagation Performance:
  Average: 782.01 tokens/sec
  P99:     822.40 tokens/sec
  Samples: 347
  Total Tokens Consumed: 51,200

🌐 Teacher API Usage:
  Total Tokens Sent: 3,813,261
  Total Tokens Received: 4,991
  Input Tokens/sec: 1,234.56
  Output Tokens/sec: 1.62
  Average per Epoch: 540,230
  Breakdown by Epoch:
    Epoch 1: 687,218 tokens
    ...

⏱️  Training Duration:
  Total Time: 2h 15m 30s (8130.5 seconds)
```

## Data Format

Training data should be in JSONL format:
```json
{"prompt": "Implement a binary search function", "language": "python"}
{"prompt": "Create a thread-safe queue", "language": "cpp"}
{"prompt": "Write a parser for JSON", "language": "rust"}
```

## Critical Statistics

The training script prints the following statistics:

- **Loss**: Total training loss (policy + KL)
- **Policy Loss**: Reward-weighted policy gradient loss
- **KL Penalty**: Divergence from base model
- **Avg Reward**: Average reward score (0-1)
- **Reward Std**: Standard deviation of rewards
- **GPU Utilization**: GPU/Neural Engine utilization percentage
- **Epoch Summary**: Per-epoch averages

### Training Summary

At the end of training, a comprehensive summary is printed including:

#### Generation Performance
- **Average**: Average generation speed (tokens/sec)
- **P99**: 99th percentile generation speed
- **Samples**: Number of generation samples
- **Total Tokens Generated**: Cumulative tokens generated across all batches

#### Backpropagation Performance
- **Average**: Average backpropagation speed (tokens/sec)
- **P99**: 99th percentile backpropagation speed
- **Samples**: Number of backpropagation samples
- **Total Tokens Consumed**: Cumulative tokens processed during backpropagation

#### Teacher API Usage
- **Total Tokens Sent**: Total input tokens sent to teacher API
- **Total Tokens Received**: Total output tokens received from teacher API
- **Input Tokens/sec**: Throughput for input tokens
- **Output Tokens/sec**: Throughput for output tokens
- **Average per Epoch**: Average tokens per epoch
- **Breakdown by Epoch**: Token usage per epoch

#### Training Duration
- **Total Time**: Complete training duration in hours, minutes, and seconds

## TensorBoard Metrics

This section provides a comprehensive reference for all metrics logged to TensorBoard during training.

### Training Metrics (`Train/`)

- **`Train/Loss`**: Total training loss (policy loss + KL penalty)
- **`Train/PolicyLoss`**: Policy gradient loss
- **`Train/KLPenalty`**: KL divergence penalty
- **`Train/AvgReward`**: Average reward for the batch
- **`Train/RewardStd`**: Standard deviation of rewards (consistency measure)
- **`Train/RewardMin`**: Minimum reward in batch
- **`Train/RewardMax`**: Maximum reward in batch
- **`Train/RewardVariance`**: Variance of rewards (lower = more consistent)

### Epoch Metrics (`Epoch/`)

- **`Epoch/AvgReward`**: Average reward across the epoch
- **`Epoch/AvgLoss`**: Average loss across the epoch
- **`Epoch/RewardVariance`**: Reward variance across the epoch (lower = more consistent)
- **`Epoch/RewardTrend`**: Change in reward from previous epoch (positive = improving)
- **`Epoch/LossTrend`**: Change in loss from previous epoch (positive = improving, as loss should decrease)
- **`Epoch/APITokens`**: Total API tokens sent to teacher model this epoch
- **`Epoch/NumSamples`**: Number of training samples processed this epoch

### Performance Metrics (`Performance/`)

#### Generation Performance
- **`Performance/Generation_TokensPerSec`**: Real-time generation speed (logged every 10 batches)
- **`Performance/Generation_Time`**: Time taken for generation (logged every 10 batches)
- **`Performance/Generation_AvgTokensPerSec`**: Average generation speed across all epochs
- **`Performance/Generation_P99TokensPerSec`**: P99 generation speed (99th percentile)
- **`Performance/Generation_TotalTokens_Cumulative`**: Cumulative total tokens generated

#### Backpropagation Performance
- **`Performance/Backprop_TokensPerSec`**: Real-time backpropagation speed (logged at logging steps)
- **`Performance/Backprop_AvgTokensPerSec`**: Average backpropagation speed across all epochs
- **`Performance/Backprop_P99TokensPerSec`**: P99 backpropagation speed (99th percentile)
- **`Performance/Training_TotalTokens_Cumulative`**: Cumulative total tokens processed during training

#### Scoring Performance
- **`Performance/Scoring_TotalTokens_Input_Cumulative`**: Cumulative input tokens for scoring
- **`Performance/Scoring_TotalTokens_Output_Cumulative`**: Cumulative output tokens for scoring

### Scoring Metrics (`Scoring/`)

These represent the scoring weights used for evaluation:
- **`Scoring/Correctness`**: Correctness weight (0.3)
- **`Scoring/CodeQuality`**: Code quality weight (0.3)
- **`Scoring/Efficiency`**: Efficiency weight (0.2)
- **`Scoring/Documentation`**: Documentation weight (0.2)

*Note: Currently these are fixed weights. Future versions may track actual criterion scores.*

### System Metrics (`System/`)

#### CPU and Memory
- **`System/CPU_Percent`**: Overall CPU usage percentage
- **`System/Memory_Percent`**: System memory usage percentage
- **`System/Memory_Used_GB`**: System memory used (GB)
- **`System/Memory_Available_GB`**: System memory available (GB)
- **`System/Process_Memory_GB`**: Process RSS (GB) (**back-compat** tag; same as `System/Process_RSS_GB`)
- **`System/Process_RSS_GB`**: Process RSS (GB)
- **`System/Process_Footprint_GB`**: **macOS physical footprint** (GB) — should match **Activity Monitor “Memory”** more closely

#### GPU/MPS Metrics
- **`System/GPU_Memory_Used_GB`**: GPU/MPS memory used (GB)
- **`System/GPU_Memory_Total_GB`**: Total GPU/MPS memory (GB)
- **`System/GPU_Memory_Percent`**: GPU/MPS memory usage percentage
- **`System/GPU_Utilization`**: **Estimated** GPU utilization (%). On Apple Silicon this is often a **proxy** (see below).
- **`System/GPU_Utilization_Estimated`**: Same value as above, but explicitly named to avoid confusion.

Note on “Metal Counter API”: it’s designed to profile **Metal command buffers you create** (e.g., a Metal app you control) using counter sample buffers, not to query arbitrary GPU counters for other frameworks’ internal Metal work. For MLX / PyTorch MPS, you generally can’t attach Metal counters from Python to their internal command streams. For deep GPU profiling use Xcode Instruments / Metal System Trace or build a dedicated Metal workload that uses counters (see: `https://developer.apple.com/videos/play/tech-talks/10001/`).

### Metal fragmentation / cache proxies (recommended)

On Apple Silicon, “OOM” can happen due to Metal heap pressure/fragmentation even when system memory looks available. This repo logs **best-effort proxies** per batch:

- **`Batch/Metal/MPS_Fragmentation_GB`**: \( \text{driver\_allocated} - \text{current\_allocated} \) from PyTorch MPS (cache/fragmentation proxy)
- **`Batch/Metal/MLX_Cache_GB`**: MLX Metal cache size (proxy for retained allocations)
- **`Batch/Metal/*_Growth_GB`**: growth since previous batch
- **`Batch/Metal/GC_Triggered`**: whether the trainer triggered cleanup

HealthCheck can trigger cleanup automatically (config under `logging.*health_check_fragmentation*`):

- `gc.collect()`
- `torch.mps.empty_cache()`
- `mx.metal.clear_cache()`

System metrics are logged:
- At each logging step (alongside training metrics)
- Continuously in the background every 5 seconds

### Viewing Metrics in TensorBoard

1. Start TensorBoard:
   ```bash
   # Suppress pkg_resources deprecation warning
   PYTHONWARNINGS=ignore::UserWarning uv run tensorboard --logdir ./logs/tensorboard
   ```

2. Open in browser:
   - Navigate to `http://localhost:6006`
   - Select metrics from the left sidebar
   - Use the time range selector to focus on specific epochs

### Offline JSON summaries (batch + epoch)

If you want to analyze results offline (or diff runs across machines), enable:

- `logging.save_json_summaries: true`
- `logging.json_summaries_dir: ./logs/json_summaries`

This will write:

- `./logs/json_summaries/batches.jsonl`
- `./logs/json_summaries/epochs.jsonl`

View as a time series (and optionally export CSV):

```bash
# Last 20 batches
uv run python scripts/visualization/view_json_summaries.py --dir ./logs/json_summaries --type batch --tail 20

# All epochs
uv run python scripts/visualization/view_json_summaries.py --dir ./logs/json_summaries --type epoch --tail 0

# Export batches to CSV
uv run python scripts/visualization/view_json_summaries.py --dir ./logs/json_summaries --type batch --csv-out ./logs/json_summaries/batches.csv
```

### Training health checks (recommended)

The trainer can emit periodic **HealthCheck** logs while training is running to detect regressions early.

Config:

- `logging.health_check_enabled: true`
- `logging.health_check_interval_batches: 5`

It prints a compact line like:

- `[HealthCheck] OK | e1 b10 | reward(mean=0.712, ema=0.690, bestN=0.801) | time(gen=78%, score=6%, train=16%) | gen(tok/s raw=5.6, kept=4.7, div=0.83)`

If unhealthy, it prints `WARN` with the specific issues (e.g. generation bottleneck, low diversity, no gain vs baseline).

### Key Metrics to Monitor

#### Training Health
- **`Epoch/AvgReward`**: Should increase over epochs (target: 0.7+)
- **`Epoch/AvgLoss`**: Should decrease and stabilize
- **`Epoch/RewardVariance`**: Should be low (< 0.01) for consistent training

#### Performance Bottlenecks
- **`Performance/Generation_TokensPerSec`**: If < 1.0, consider enabling MLX
- **`Performance/Backprop_TokensPerSec`**: Monitor for training efficiency

#### System Resources
- **`System/GPU_Utilization_Estimated`**:
  - If `logging.gpu_utilization_mode: "memory_proxy"` (default): treat this as a **rough signal only** (it may not match Activity Monitor, especially when MLX is generating).
  - If `logging.gpu_utilization_mode: "powermetrics"`: should track Activity Monitor more closely (best-effort; may require elevated permissions).
- **`System/Memory_Percent`**: Monitor for OOM warnings (> 90%)

#### Convergence Indicators
- **`Epoch/RewardTrend`**: Should be positive and decreasing (approaching 0)
- **`Epoch/LossTrend`**: Should be positive (loss decreasing) and stabilizing

### Metric Groups

#### Reward Analysis
- `Epoch/AvgReward`
- `Epoch/RewardTrend`
- `Epoch/RewardVariance`
- `Train/RewardStd`
- `Train/RewardVariance`

#### Loss Analysis
- `Epoch/AvgLoss`
- `Epoch/LossTrend`
- `Train/Loss`
- `Train/PolicyLoss`
- `Train/KLPenalty`

#### Performance Analysis
- `Performance/Generation_AvgTokensPerSec`
- `Performance/Generation_P99TokensPerSec`
- `Performance/Backprop_AvgTokensPerSec`
- `Performance/Backprop_P99TokensPerSec`

#### System Health
- `System/CPU_Percent`
- `System/Memory_Percent`
- `System/GPU_Utilization_Estimated`
- `System/GPU_Memory_Percent`

## Performance Optimizations

### M5 MacBook Optimizations

The script is optimized for M5 MacBook:

- **4-bit Quantization**: Reduces memory usage
- **MPS Backend**: Uses Metal Performance Shaders
- **Bfloat16**: Efficient mixed precision training
- **Gradient Accumulation**: Simulates larger batch sizes
- **Small Batch Size**: Fits in M5 memory constraints
- **MLX Format**: Models are automatically saved in MLX format for faster inference on Apple Silicon
- **Gradient Checkpointing**: Trades compute for memory to prevent OOM
- **Aggressive Cache Clearing**: Prevents MPS memory buildup

### Model Loading Optimizations

#### Fast Model Loading
- **Safetensors Format**: Uses safetensors for faster checkpoint loading (up to 2x faster)
- **Low CPU Memory Usage**: Optimizes memory during loading
- **Fast Tokenizer**: Uses fast tokenizer implementation
- **Progress Tracking**: Shows loading time for monitoring with detailed profiling
- **Memory Cache Clearing**: Clears MPS/CUDA cache before/after loading
- **Memory Monitoring**: Real-time memory usage tracking during loading

#### Preload Model (Recommended)
To avoid slow loading on every training run, preload the model once:

```bash
uv run python scripts/utils/preload_model.py --model Qwen/Qwen2.5-Coder-3B-Instruct
```

This will:
- Download and cache the model locally
- Verify the model works correctly
- Make subsequent training runs start much faster

**First load**: ~5-10 minutes (download + load)  
**Subsequent loads**: ~1-2 minutes (from cache)

#### Understanding Loading Times

For Qwen2.5-Coder-3B-Instruct with 4-bit quantization:

- **Tokenizer Loading**: ~1-2 seconds
- **Model Config**: ~0.5 seconds
- **Checkpoint Shards** (4 shards):
  - Shard 1 (0-25%): ~15-20 seconds
  - Shard 2 (25-50%): ~15-20 seconds
  - Shard 3 (50-75%): ~15-20 seconds
  - Shard 4 (75-100%): ~25-35 seconds (slower)

The last shard is slower due to:
1. Memory pressure from previous shards
2. Device mapping finalization
3. Quantization setup completion
4. Model initialization

### Training Loop Optimizations

#### Key Optimizations Applied

1. **DataLoader Configuration**
   - `num_workers=0`: M5's unified memory architecture doesn't benefit from multiple workers
   - `pin_memory=False`: Not needed for M5
   - Eliminates tokenizer fork warnings

2. **Batch Processing**
   - Batch student generation: Generate multiple samples in batches
   - Reduced max_new_tokens: 64 instead of 128 for faster generation
   - Batch tokenization: Process all prompts together

3. **Parallel API Calls**
   - ThreadPoolExecutor: Process teacher API calls concurrently (up to 4 workers)
   - Caching: Cache teacher responses to avoid redundant API calls
   - Async processing: Collect results as they complete

4. **Reduced Sample Count**
   - `num_samples_per_prompt: 2` (reduced from 4)
   - Significantly reduces API calls and generation time

5. **Memory Optimizations**
   - Reduced max_length: 512 instead of 1024
   - Smaller generation batches: Process in chunks
   - Efficient tensor operations: Use bfloat16 throughout
   - Aggressive cache clearing: Clear MPS cache every batch

6. **Tokenizer Parallelism**
   - `TOKENIZERS_PARALLELISM=false`: Prevents fork warnings on M5
   - Set as environment variable at module level

### Configuration Recommendations

For M5 with 32GB:

```yaml
training:
  batch_size: 2  # Reduced to prevent OOM
  gradient_accumulation_steps: 16  # Increased to maintain effective batch size

rlaif:
  num_samples_per_prompt: 2  # Reduced from 4

model:
  max_length: 512  # Reduced from 1024 to prevent OOM

hardware:
  dataloader_num_workers: 0  # Critical for M5
  use_mlx_for_generation: true  # Enable for 5-10x faster generation
  mlx_model_path: "./mlx_model/q8"
  mlx_quantization: q8_bit
```

## MLX Optimization (5-10x Faster Generation)

### Problem

PyTorch MPS (Metal Performance Shaders) on Apple Silicon provides GPU acceleration, but generation is still slow (~0.2 tokens/sec). This is because:
- PyTorch MPS doesn't fully utilize Apple's Neural Engine
- MPS has overhead from CPU-GPU synchronization
- Unified memory architecture benefits from native frameworks

### Solution: MLX Framework

**MLX** is Apple's native machine learning framework optimized for Apple Silicon. It provides:
- **5-10x faster inference** than PyTorch MPS
- **Better GPU utilization** (uses both GPU and Neural Engine)
- **Lower memory overhead** on unified memory
- **Native Apple Silicon optimization**

### Quick Start

#### 1. Convert Model to MLX Format

```bash
# Preferred (per Qwen docs): use mlx-lm's CLI converter
# Ref: https://qwen.readthedocs.io/en/latest/run_locally/mlx-lm.html

# Non-quantized
uv run mlx_lm.convert --hf-path Qwen/Qwen2.5-Coder-3B-Instruct --mlx-path ./mlx_model/base

# Quantized (recommended): Q4 or Q8
uv run mlx_lm.convert --hf-path Qwen/Qwen2.5-Coder-3B-Instruct --mlx-path ./mlx_model/q4 -q --q-bits 4
uv run mlx_lm.convert --hf-path Qwen/Qwen2.5-Coder-3B-Instruct --mlx-path ./mlx_model/q8 -q --q-bits 8

# Alternatively, you can use our wrapper (calls mlx_lm.convert under the hood)
uv run python scripts/utils/convert_to_mlx.py --hf-path Qwen/Qwen2.5-Coder-3B-Instruct --mlx-path ./mlx_model/q8 --quantize q8_bit
```

#### 2. Update Config

Edit `config.yaml`:

```yaml
hardware:
  use_mlx_for_generation: true  # Enable MLX for generation
  mlx_model_path: "./mlx_model/q8"  # Path to MLX model (e.g., ./mlx_model/q8, ./mlx_model/q4, ./mlx_model/base)
  mlx_quantization: q8_bit  # Options: q4_bit, q8_bit, or null
```

#### 3. Run Training

```bash
uv run python scripts/training/train_rlaif.py --config config.yaml
```

The training will now use MLX for generation (5-10x faster) while keeping PyTorch for training.

#### Keeping generation weights in sync with training updates

Today this repo uses:

- **PyTorch** for backprop/training (weight updates)
- **MLX** for generation (fast inference)

These are different runtimes, so there is **no cheap per-batch weight sharing** between them.

Best-effort sync option:

- Set `hardware.reload_mlx_from_latest_checkpoint: true` (default)
- Save checkpoints frequently (`training.save_every_batches` or `training.save_every_epochs`)

When a checkpoint is saved in MLX format, the trainer will **reload the MLX generation model from that checkpoint** so subsequent batches generate with fresher weights. Beware: converting/reloading MLX weights is expensive if done too frequently.

### MLX generation worker mode (recommended on Apple Silicon)

If you run **MLX generation** + **PyTorch MPS training** in the same Python process, their Metal allocators can fight and you may see Metal command-buffer OOMs due to heap pressure/fragmentation.

This repo supports running MLX generation in a **separate subprocess**:

- PyTorch/MPS training stays in the main process
- MLX/Metal generation runs in `scripts/utils/mlx_gen_worker.py`

Enable in `config.yaml`:

- `hardware.use_mlx_generation_worker: true`
- `hardware.mlx_generation_worker_timeout_s: 240`

Troubleshooting:
- If you see `Broken pipe`, it means the worker process exited. The trainer now auto-restarts the worker once and will surface the worker stderr tail on failure. Common causes:
  - invalid `hardware.mlx_model_path` (worker can't `mlx_lm.load()` the directory)
  - missing `mlx` / `mlx-lm` in the active `uv` environment

Notes:
- This isolates MLX allocations from the training process (often stabilizes long runs).
- Generation semantics match the in-process MLX path (greedy by default, sampling on retries).

### Performance Comparison

#### PyTorch MPS (Baseline)
- **Speed**: ~0.2 tokens/sec
- **Generation time**: ~89s for 20 tokens
- **Uses**: GPU only (MPS)

#### MLX (Optimized)
- **Speed**: ~2-3 tokens/sec (10-15x faster)
- **Generation time**: ~10-20s for 20 tokens
- **Uses**: GPU + Neural Engine

### Quantization Options

#### 4-bit Quantization (q4_bit)
- **Size**: ~4GB (vs ~14GB full precision)
- **Speed**: ~2.8 tokens/sec (similar to Q8, but uses less memory)
- **Quality**: Very good (slight degradation)
- **Use case**: Memory-constrained systems, fastest inference

#### 8-bit Quantization (q8_bit)
- **Size**: ~7GB
- **Speed**: ~2.7-2.8 tokens/sec
- **Quality**: Excellent
- **Use case**: Best balance (recommended)

#### No Quantization
- **Size**: ~14GB
- **Speed**: ~1-2 tokens/sec
- **Quality**: Best
- **Use case**: Maximum quality

### How It Works

#### Architecture

```
Training Loop:
├── Generation (MLX) ← Fast inference using GPU + Neural Engine
├── Scoring (Teacher API) ← Parallel API calls
└── Training (PyTorch) ← Gradient updates with MPS
```

#### Implementation

1. **Dual Model Setup**:
   - PyTorch model: Used for training (gradient updates)
   - MLX model: Used for generation (inference only)

2. **Automatic Fallback**:
   - If MLX model not available, falls back to PyTorch MPS
   - Seamless transition, no code changes needed

3. **Memory Management**:
   - MLX uses less memory than PyTorch
   - Better for unified memory architecture

### Using MLX Models

```python
from mlx_lm import load, generate

# Load MLX model
model, tokenizer = load("./checkpoints/checkpoint-500/mlx_model")

# Generate code
prompt = "Write high-quality python code:\n\nImplement binary search\n\nCode:"
response = generate(model, tokenizer, prompt=prompt, max_tokens=512)
print(response)
```

Or use the utility script:
```bash
uv run python scripts/utils/load_mlx_model.py \
    --model_path ./checkpoints/checkpoint-500/mlx_model \
    --prompt "Implement a binary search function" \
    --language python
```

### Performance Tips

1. **Pre-convert models**: Convert once, use many times
2. **Use quantization**: 8-bit is best balance
3. **Warm up MLX**: First generation is slower (warmup included)
4. **Monitor tokens/sec**: Should see 2-3 tokens/sec with MLX
5. **Reduce max_tokens**: Shorter generations = faster

### Troubleshooting

#### MLX Model Not Found
If you see "MLX model not found", the code will fall back to PyTorch MPS. To fix:

1. Convert model to MLX:
   ```bash
   uv run python scripts/utils/convert_to_mlx.py --hf-path Qwen/Qwen2.5-Coder-3B-Instruct --mlx-path ./mlx_model
   ```

2. Update config:
   ```yaml
   hardware:
     use_mlx_for_generation: true
     mlx_model_path: "./mlx_model/q8"
   ```

#### Slow Generation Still
If generation is still slow even with MLX:

1. **Check MLX is being used**: Look for "Generating with MLX" in logs
2. **Reduce batch size**: Smaller batches = faster per-sample
3. **Use quantization**: 4-bit or 8-bit quantization speeds up inference
4. **Check memory**: Ensure enough free memory (MLX needs less than PyTorch)

## Model Validation

After training, validate your model to compare **pre-training (baseline model)** vs **post-training (fine-tuned model)** quality.

On Apple Silicon, validation is fastest and most reliable when you run **both models in MLX** (no multi-GB Hugging Face downloads, and no PyTorch/MPS generation).

```bash
uv run python scripts/validation/validate_model.py \
    --fine_tuned_path ./checkpoints/checkpoint-500 \
    --teacher_provider anthropic \
    --baseline_mlx_path ./mlx_model/q4 \
    --max_samples 5 \
    --print_chars 500 \
    --output ./validation_results.json
```

The validation script will:
- Generate code from both baseline and fine-tuned models (**MLX by default**)
- Score outputs using the teacher model
- Show improvement statistics
- Display example comparisons
- Print **generation throughput (tokens/sec)** for both baseline and fine-tuned outputs

Notes:
- By default, the fine-tuned MLX model is loaded from `<fine_tuned_path>/mlx_model`.
- If the checkpoint MLX export is incompatible (older config/weights mismatch), the script will automatically rebuild it into `<fine_tuned_path>/mlx_model_repaired` using `mlx_lm.convert` and continue.

### Example Output

```
================================================================================
VALIDATION REPORT: Pre-Training(baseline model) vs Post-Training (fine tuned model)
================================================================================

Test Cases: 5

Average Reward Scores:
  Baseline Model:    0.6234
  Fine-tuned Model:  0.7891
  Average Improvement: +0.1657 (+26.58%)

Improvement Distribution:
  Improved:  4 cases
  Degraded:  1 cases
  Equal:     0 cases
```

### Custom Test Prompts

Create a JSONL file with your test cases:

```json
{"prompt": "Implement a binary search function", "language": "python"}
{"prompt": "Create a thread-safe queue", "language": "cpp"}
{"prompt": "Write a JSON parser", "language": "rust"}
```

Then run:
```bash
uv run python scripts/validation/validate_model.py \
    --fine_tuned_path ./checkpoints/checkpoint-500 \
    --teacher_provider anthropic \
    --baseline_mlx_path ./mlx_model/q4 \
    --test_prompts ./my_test_prompts.jsonl \
    --max_samples 5
```

## Dataset Collection and Upload

During training, the system automatically collects:
- **Prompts**: Original code generation prompts
- **Teacher Code**: High-quality reference code from teacher model
- **Student Code**: Code generated by the student model
- **Scores**: Quality scores for both student and teacher code
- **Rewards**: Normalized reward values
- **Scoring Parameters**: Detailed scoring breakdown

These are saved as datasets and can be uploaded to Hugging Face for sharing and reuse.

### Automatic Dataset Collection

Datasets are automatically collected during training and saved to `./datasets/`:
- `train.jsonl`: Training examples
- `validation.jsonl`: Validation examples  
- `evaluation.jsonl`: Evaluation examples
- `README.md`: Dataset card with metadata

### Data Structure

Each dataset entry contains:

```json
{
  "prompt": "Implement a binary search function",
  "language": "python",
  "student_code": "def binary_search(arr, target):\n    ...",
  "teacher_code": "def binary_search(arr, target):\n    ...",
  "student_score": 0.85,
  "teacher_score": 0.92,
  "reward": 0.924,
  "scoring_breakdown": {
    "correctness": 0.3,
    "code_quality": 0.3,
    "efficiency": 0.2,
    "documentation": 0.2
  },
  "timestamp": "2025-12-18T10:30:00"
}
```

### Upload Datasets to Hugging Face

Configure in `config.yaml`:
```yaml
huggingface:
  upload_datasets: true
  dataset_repo_id: "mlx-community/code-rlaif-dataset"
  save_datasets_locally: true
  dataset_output_dir: "./datasets"
```

Datasets are automatically uploaded after training completes.

### Manual Dataset Export

Export and upload datasets manually:
```bash
uv run python scripts/utils/export_datasets.py \
    --dataset_dir ./datasets \
    --repo_id mlx-community/code-rlaif-dataset \
    --hf_token $HUGGINGFACE_TOKEN
```

### Using Uploaded Datasets

```python
from datasets import load_dataset

# Load from Hugging Face
dataset = load_dataset("mlx-community/code-rlaif-dataset")

# Access training data
train_data = dataset['train']
print(train_data[0])
# {
#   'prompt': 'Implement binary search',
#   'language': 'python',
#   'student_code': '...',
#   'teacher_code': '...',
#   'student_score': 0.85,
#   'teacher_score': 0.92,
#   'reward': 0.924,
#   ...
# }
```

## Hugging Face Model Upload

Upload your fine-tuned MLX model to the [MLX Community](https://huggingface.co/mlx-community) on Hugging Face:

1. Set your Hugging Face token:
   ```bash
   export HUGGINGFACE_TOKEN="hf_..."
   ```

2. Configure upload in `config.yaml`:
   ```yaml
   huggingface:
     upload_to_hub: true
     repo_id: "mlx-community/code-rlaif"
     hf_token_env: "HUGGINGFACE_TOKEN"
     upload_quantized: true
     private: false
   ```

3. The model will be automatically uploaded after training completes or at specified checkpoints.

The uploaded model will include:
- MLX format weights
- Model card with training details
- Usage examples
- Training statistics

## Profiling and Debugging

### Profiling Model Loading

To profile model loading and training with call stacks and timing:

```bash
# Profile with Apple Instruments (recommended for macOS)
uv run python scripts/profiling/profile_with_instruments.py --method instruments --script scripts/profiling/profile_model_loading.py

# Profile with py-spy (flamegraph)
uv run python scripts/profiling/profile_with_instruments.py --method pyspy --script scripts/profiling/profile_model_loading.py

# Profile with all methods
uv run python scripts/profiling/profile_with_instruments.py --method all --script scripts/profiling/profile_model_loading.py
```

### Available Profiling Methods

1. **Apple Instruments** (Recommended for macOS)
   - Native Apple profiler with GUI
   - Call stack visualization with time spent in each function
   - Memory allocations and leaks
   - CPU usage per function
   - Thread analysis

2. **py-spy** - Python call stack profiler with flamegraphs
   - Low-overhead profiling
   - Beautiful flamegraphs
   - Works without Xcode

3. **cProfile** - Python's built-in profiler
   - Detailed function-level statistics
   - Can be visualized with snakeviz

4. **memory_profiler** - Memory usage profiling
   - Line-by-line memory usage
   - Memory increment tracking

### Quick Reference

```bash
# Instruments (Time Profiler)
instruments -t "Time Profiler" -D trace.trace python script.py

# py-spy (Flamegraph)
py-spy record -o flamegraph.svg -- python script.py

# cProfile
python -m cProfile -o profile.prof script.py
snakeviz profile.prof

# Memory Profiler
python -m memory_profiler script.py
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size` to 2 in config
- Increase `gradient_accumulation_steps` to 16
- Enable `use_4bit: true`
- Reduce `max_length` to 512
- Enable gradient checkpointing (already enabled)
- Clear MPS cache more frequently (already done)

### Slow Training

- **Enable MLX**: Convert model to MLX for 5-10x faster generation
- Reduce `num_samples_per_prompt` to 2
- Use smaller teacher model (GPT-3.5 instead of GPT-4)
- Reduce `max_length` to 512
- Reduce `max_new_tokens` in generation to 64

### API Rate Limits

- Add delays between API calls
- Use smaller batches
- Consider caching teacher responses (already implemented)

### NaN Loss

The code now includes comprehensive NaN detection and handling:
- Validates all inputs (logits, rewards, log_probs)
- Replaces NaN/Inf with safe values
- Clamps rewards to [0, 1]
- Handles shape mismatches
- Falls back to zero loss if NaN persists

If you still see NaN:
- Check reward values (should be 0-1)
- Check for invalid prompts or code
- Verify teacher model is working correctly

### Generation Bottleneck

If generation is taking >70% of time:
- **Enable MLX**: This is the most important optimization
- Reduce `max_new_tokens` to 64
- Use Q4 or Q8 quantization
- Reduce `num_samples_per_prompt` to 2

### Model Loading Issues

#### If Loading is Very Slow (>10 minutes from cache)

1. **Check Disk I/O**: Slow disk can cause slow loading
2. **Check Memory**: Insufficient memory causes swapping
3. **Use Safetensors**: Ensure safetensors is installed
4. **Clear Cache**: Sometimes corrupted cache causes issues

#### If You See Warnings

- **`torch_dtype` deprecation**: Already fixed in latest code
- **Generation flags**: Already fixed in latest code
- **Tokenizer parallelism**: Suppressed with `TOKENIZERS_PARALLELISM=false`
- **PyTorch inductor warnings**: Suppressed (harmless on MPS)

### Validation Issues

**"Teacher model not available"**
- Set API keys: `export OPENAI_API_KEY="..."` or `export ANTHROPIC_API_KEY="..."`
- The script will use fallback scoring if teacher is unavailable

**"Out of memory"**
- Use smaller test sets
- Reduce `max_tokens` in generation

### Upload Issues

**"Authentication failed"**
- Check your `HUGGINGFACE_TOKEN` is set correctly
- Verify token has write permissions

**"Repository not found"**
- Ensure you've joined the MLX Community organization
- Check repository name is correct (format: `mlx-community/model-name`)

**"Upload timeout"**
- Large models may take time to upload
- Check your internet connection
- Try uploading during off-peak hours

## Project Structure

```
train_coding_asst/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Training configuration
├── .gitignore               # Git ignore rules
│
├── scripts/
│   ├── training/              # Training scripts
│   │   ├── train_rlaif.py     # Main RLAIF training script
│   │   └── train_rfai.py      # Legacy RFAI script (deprecated)
│   ├── validation/            # Validation scripts
│   │   └── validate_model.py  # Model validation and comparison
│   ├── utils/                 # Utility scripts
│   │   ├── data_utils.py      # Dataset generation utilities
│   │   ├── convert_to_mlx.py  # MLX model conversion
│   │   ├── preload_model.py   # Model preloading and caching
│   │   ├── load_mlx_model.py # MLX model loader
│   │   └── export_datasets.py # Dataset export to Hugging Face
│   ├── profiling/            # Performance profiling scripts
│   │   ├── profile_model_loading.py  # Model loading profiler
│   │   └── profile_with_instruments.py  # Apple Instruments profiler
│   └── visualization/        # Visualization scripts
│       └── visualize_training.py  # Training visualization utilities
├── run_training.sh          # Convenience script to run training
│
├── mlx_model/               # Consolidated MLX model directory
│   ├── base/                # Unquantized base model
│   ├── q4/                   # Q4 quantized model
│   └── q8/                   # Q8 quantized model (recommended)
│
├── checkpoints/             # Model checkpoints (created during training)
│   └── checkpoint-{step}/
│       ├── config.json
│       ├── model-*.safetensors
│       ├── tokenizer files
│       ├── training_stats.json
│       └── mlx_model/       # MLX format (if enabled)
│
├── logs/                    # Training logs (created during training)
│   └── tensorboard/
│       └── events.out.tfevents.*
│
└── datasets/                # Collected datasets (created during training)
    ├── train.jsonl
    ├── validation.jsonl
    ├── evaluation.jsonl
    └── README.md
```

## Output Structure

```
train_coding_asst/
├── checkpoints/
│   └── checkpoint-{step}/
│       ├── config.json
│       ├── model-*.safetensors      # PyTorch format
│       ├── tokenizer files
│       ├── training_stats.json
│       └── mlx_model/               # MLX format (for Apple Silicon)
│           ├── config.json
│           ├── model-*.safetensors
│           ├── tokenizer files
│           └── README.md
├── logs/
│   └── tensorboard/
│       └── events.out.tfevents.*
└── data/
    ├── train.jsonl
    └── eval.jsonl
```

## Advanced Usage

### Custom Reward Function
Modify `TeacherModel.score_code()` to implement custom scoring logic.

### Multi-Language Training
The system supports Python, C++, and Rust. Add more languages by:
1. Adding language to config
2. Updating prompt formatting
3. Ensuring teacher model supports the language

### Fine-tuning Specific Aspects
Adjust reward weights in teacher scoring to emphasize:
- Correctness: Increase correctness weight
- Code quality: Increase quality weight
- Efficiency: Increase efficiency weight

## Training Time Estimates

- **Small dataset (100 samples)**: ~2-4 hours
- **Medium dataset (1000 samples)**: ~1-2 days
- **Large dataset (10000 samples)**: ~1-2 weeks

*Times vary based on API response times, model size, and whether MLX is enabled*

## References

- [Qwen Models](https://github.com/QwenLM/Qwen)
- [Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2009.01325)
- [PPO for Language Models](https://arxiv.org/abs/2009.01325)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM Documentation](https://github.com/ml-explore/mlx-examples)
- [Apple Neural Engine](https://developer.apple.com/machine-learning/neural-engine/)
- [Apple Instruments User Guide](https://developer.apple.com/documentation/instruments)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [MLX Community](https://huggingface.co/mlx-community)

## License

MIT License
