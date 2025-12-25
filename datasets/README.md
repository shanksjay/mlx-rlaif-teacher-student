---
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

This dataset contains code generation prompts and their associated programming languages from the RLAIF (Reinforcement Learning from AI Feedback) training process.

## Dataset Description

This dataset was generated during the fine-tuning of a Qwen model for code generation using RLAIF methodology. Each entry includes:

- **Prompt**: The code generation prompt/instruction
- **Language**: Programming language (python, cpp, rust)

## Dataset Structure

- **Training Set**: 2800 examples
- **Validation Set**: 0 examples
- **Evaluation Set**: 0 examples

## Data Fields

Each example contains:
- `prompt` (string): Code generation prompt/instruction
- `language` (string): Programming language (python, cpp, rust)

## Usage

### Load from Hugging Face

```python
from datasets import load_dataset

dataset = load_dataset("mlx-community/code-rlaif-dataset")

# Access training data
train_data = dataset['train']
print(train_data[0])
```

### Load from Local Files

```python
import json

# Load training data
# Each line is a JSON object with format: {"prompt": "instruction", "language": "language"}
with open('datasets/train.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line)
        # Entry format: {"prompt": "instruction", "language": "python|cpp|rust"}
        print(f"Prompt: {entry['prompt']}")
        print(f"Language: {entry['language']}")
```

## Training Details

- **Base Model**: Qwen/Qwen2.5-Coder-3B-Instruct
- **Teacher Model**: anthropic/claude-3-5-haiku-20241022
- **Training Steps**: 21
- **Languages**: Python, C++, Rust
- **Average Reward**: 0.1867

## Scoring Methodology

Scores are computed using the teacher model evaluating code on:
- **Correctness** (30%): Does the code solve the problem correctly?
- **Code Quality** (30%): Is it clean, readable, and well-structured?
- **Efficiency** (20%): Is it efficient and follows best practices?
- **Documentation** (20%): Is it well-documented?

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{code_rlaif_dataset,
  title={Code RLAIF Dataset},
  author={MLX Community},
  year={2024},
  url={https://huggingface.co/datasets/mlx-community/code-rlaif-dataset}
}
```

## License

MIT License
