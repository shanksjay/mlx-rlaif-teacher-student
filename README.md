# Qwen Code Fine-tuning with RFAI (Reinforcement from AI Feedback)

This project implements a teacher-student training scheme to fine-tune a Qwen model for generating high-quality C++, Python, and Rust code using Reinforcement from AI Feedback (RFAI).

## Overview

The training system uses a two-model architecture:
- **Teacher Model**: OpenAI Codex (GPT-4) or Claude - provides high-quality reference code and scoring
- **Student Model**: Qwen model (Qwen2.5-7B-Instruct) - being fine-tuned to match teacher quality

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    RFAI Training Pipeline                       │
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
pip install -r requirements.txt
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
python data_utils.py
```

## Configuration

Edit `config.yaml` to customize training:

- **Model**: Choose base Qwen model, quantization settings
- **Teacher**: Select OpenAI or Anthropic, model variant
- **Training**: Batch size, learning rate, epochs, etc.
- **RFAI**: Reward weights, KL penalty, sampling parameters
- **Hardware**: MPS settings for M5 MacBook

## Usage

### Basic Training

```bash
python train_rfai.py --config config.yaml
```

### Custom Data Files

```bash
python train_rfai.py \
    --config config.yaml \
    --train_file ./data/my_train.jsonl \
    --eval_file ./data/my_eval.jsonl
```

### Monitor Training

In a separate terminal, start TensorBoard:
```bash
uv run tensorboard --logdir ./logs/tensorboard
```

Then open http://localhost:6006 in your browser.

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
- **Epoch Summary**: Per-epoch averages

## TensorBoard Metrics

The following metrics are logged to TensorBoard:

### Training Metrics
- `Train/Loss`: Total training loss
- `Train/PolicyLoss`: Policy gradient component
- `Train/KLPenalty`: KL divergence penalty
- `Train/AvgReward`: Average reward per step
- `Train/RewardStd`: Reward distribution spread
- `Epoch/AvgReward`: Per-epoch reward average
- `Epoch/AvgLoss`: Per-epoch loss average

### System Metrics
- `System/CPU_Percent`: Overall CPU usage percentage
- `System/Memory_Percent`: System memory usage percentage
- `System/Memory_Used_GB`: System memory used (GB)
- `System/Memory_Available_GB`: Available system memory (GB)
- `System/Process_Memory_GB`: Training process memory usage (GB)
- `System/GPU_Memory_Used_GB`: GPU/MPS memory used (GB, if available)
- `System/GPU_Memory_Total_GB`: Total GPU/MPS memory (GB, if available)
- `System/GPU_Memory_Percent`: GPU/MPS memory usage percentage (if available)

System metrics are logged:
- At each logging step (alongside training metrics)
- Continuously in the background every 5 seconds

## Output Structure

```
train_coding_asst/
├── checkpoints/
│   └── checkpoint-{step}/
│       ├── config.json
│       ├── pytorch_model.bin          # PyTorch format
│       ├── tokenizer files
│       ├── training_stats.json
│       └── mlx_model/                 # MLX format (for Apple Silicon)
│           ├── config.json
│           ├── model.safetensors
│           ├── tokenizer files
│           └── README_MLX.md
├── logs/
│   └── tensorboard/
│       └── events.out.tfevents.*
└── data/
    ├── train.jsonl
    └── eval.jsonl
```

## MLX Model Format

Models are automatically saved in MLX format alongside PyTorch checkpoints. MLX format provides:

- **Faster Inference**: Optimized for Apple Silicon (M5 MacBook)
- **Lower Memory Usage**: Efficient weight storage
- **Quantization Support**: Optional 4-bit or 8-bit quantization

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
python load_mlx_model.py \
    --model_path ./checkpoints/checkpoint-500/mlx_model \
    --prompt "Implement a binary search function" \
    --language python
```

## MLX Optimization (5-10x Faster Generation)

For **5-10x faster generation** on Apple Silicon, use MLX instead of PyTorch MPS:

1. **Convert model to MLX**:
   ```bash
   uv run python convert_to_mlx.py --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ./mlx_model
   ```

2. **Update config.yaml**:
   ```yaml
   hardware:
     use_mlx_for_generation: true
     mlx_model_path: "./mlx_model"
   ```

3. **Run training**: Generation will now use MLX (much faster!)

See [MLX_OPTIMIZATION_GUIDE.md](MLX_OPTIMIZATION_GUIDE.md) for detailed instructions.

## M5 MacBook Optimizations

The script is optimized for M5 MacBook:

- **4-bit Quantization**: Reduces memory usage
- **MPS Backend**: Uses Metal Performance Shaders
- **Bfloat16**: Efficient mixed precision training
- **Gradient Accumulation**: Simulates larger batch sizes
- **Small Batch Size**: Fits in M5 memory constraints
- **MLX Format**: Models are automatically saved in MLX format for faster inference on Apple Silicon

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Increase `gradient_accumulation_steps`
- Enable `use_4bit: true`
- Reduce `max_length`

### Slow Training
- Reduce `num_samples_per_prompt`
- Use smaller teacher model (GPT-3.5 instead of GPT-4)
- Reduce `max_length`

### API Rate Limits
- Add delays between API calls
- Use smaller batches
- Consider caching teacher responses

## Model Validation

After training, validate your model to compare pre-training vs post-training quality:

```bash
python validate_model.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --fine_tuned_path ./checkpoints/checkpoint-500 \
    --test_prompts ./data/eval.jsonl \
    --output ./validation_results.json
```

The validation script will:
- Generate code from both baseline and fine-tuned models
- Score outputs using the teacher model
- Show improvement statistics
- Display example comparisons

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

### Upload Datasets to Hugging Face

Configure in `config.yaml`:
```yaml
huggingface:
  upload_datasets: true
  dataset_repo_id: "mlx-community/qwen-code-rfai-dataset"
  save_datasets_locally: true
  dataset_output_dir: "./datasets"
```

Datasets are automatically uploaded after training completes.

### Manual Dataset Export

Export and upload datasets manually:
```bash
python export_datasets.py \
    --dataset_dir ./datasets \
    --repo_id mlx-community/qwen-code-rfai-dataset \
    --hf_token $HUGGINGFACE_TOKEN
```

### Using Uploaded Datasets

```python
from datasets import load_dataset

# Load from Hugging Face
dataset = load_dataset("mlx-community/qwen-code-rfai-dataset")

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
     repo_id: "mlx-community/qwen-code-rfai"
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

## Profiling

To profile model loading and training with call stacks and timing:

```bash
# Profile with Apple Instruments (recommended for macOS)
uv run python profile_with_instruments.py --method instruments --script profile_model_loading.py

# Profile with py-spy (flamegraph)
uv run python profile_with_instruments.py --method pyspy --script profile_model_loading.py

# Profile with all methods
uv run python profile_with_instruments.py --method all --script profile_model_loading.py
```

See [PROFILING_GUIDE.md](PROFILING_GUIDE.md) for detailed instructions on using Apple Instruments, py-spy, cProfile, and memory profiling tools.

## References

- [Qwen Models](https://github.com/QwenLM/Qwen)
- [Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2009.01325)
- [PPO for Language Models](https://arxiv.org/abs/2009.01325)
- [Apple Instruments User Guide](https://developer.apple.com/documentation/instruments)

## License

MIT License

