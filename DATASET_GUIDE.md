# Dataset Collection and Upload Guide

## Overview

The training system automatically collects and saves training data including:
- Sample prompts
- Teacher-generated code
- Student-generated code  
- Scoring parameters and rewards

This data is saved as datasets and can be uploaded to Hugging Face MLX Community for sharing and reuse.

## Automatic Dataset Collection

During training, the system automatically collects:

1. **Training Data**: All prompts, generated code, and scores from training
2. **Validation Data**: Validation examples (if validation dataset is provided)
3. **Evaluation Data**: Evaluation examples (if evaluation dataset is provided)

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
  "timestamp": "2024-01-15T10:30:00"
}
```

## Configuration

Configure dataset collection and upload in `config.yaml`:

```yaml
huggingface:
  upload_datasets: true  # Enable dataset upload
  dataset_repo_id: "mlx-community/qwen-code-rfai-dataset"  # Dataset repository
  save_datasets_locally: true  # Save datasets locally
  dataset_output_dir: "./datasets"  # Local save directory
```

## Local Dataset Storage

Datasets are automatically saved to the configured directory:

```
datasets/
├── train.jsonl          # Training examples
├── validation.jsonl     # Validation examples
├── evaluation.jsonl     # Evaluation examples
└── README.md            # Dataset card
```

### Dataset Format

Each file is in JSONL format (one JSON object per line):

```bash
# View training data
head -n 1 datasets/train.jsonl | jq

# Count examples
wc -l datasets/train.jsonl
```

## Uploading to Hugging Face

### Automatic Upload

Datasets are automatically uploaded after training completes if:
- `upload_datasets: true` in config
- `dataset_repo_id` is specified
- `HUGGINGFACE_TOKEN` is set

### Manual Upload

Upload datasets manually using the export script:

```bash
python export_datasets.py \
    --dataset_dir ./datasets \
    --repo_id mlx-community/qwen-code-rfai-dataset \
    --hf_token $HUGGINGFACE_TOKEN
```

### Prerequisites

1. **Hugging Face Account**: Sign up at https://huggingface.co
2. **Access Token**: Get token from https://huggingface.co/settings/tokens
3. **MLX Community Access**: Request to join https://huggingface.co/mlx-community

Set your token:
```bash
export HUGGINGFACE_TOKEN="hf_..."
```

## Using Uploaded Datasets

### Load from Hugging Face

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("mlx-community/qwen-code-rfai-dataset")

# Access splits
train_data = dataset['train']
val_data = dataset.get('validation', [])
eval_data = dataset.get('evaluation', [])

# Iterate through examples
for example in train_data:
    print(f"Prompt: {example['prompt']}")
    print(f"Language: {example['language']}")
    print(f"Student Score: {example['student_score']}")
    print(f"Teacher Score: {example['teacher_score']}")
    print(f"Reward: {example['reward']}")
    print(f"Student Code:\n{example['student_code']}")
    print(f"Teacher Code:\n{example['teacher_code']}")
    print("-" * 80)
```

### Load from Local Files

```python
import json

# Load training data
with open('datasets/train.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line)
        print(entry['prompt'])
        print(f"Score: {entry['student_score']}")
```

### Convert to Hugging Face Dataset Format

```python
from datasets import Dataset
import json

# Load from JSONL
data = []
with open('datasets/train.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Create Hugging Face dataset
dataset = Dataset.from_list(data)

# Save locally
dataset.save_to_disk('./my_dataset')

# Or push to hub
dataset.push_to_hub("mlx-community/my-dataset")
```

## Dataset Analysis

### Analyze Dataset Statistics

```python
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("mlx-community/qwen-code-rfai-dataset")
df = pd.DataFrame(dataset['train'])

# Basic statistics
print(f"Total examples: {len(df)}")
print(f"Average student score: {df['student_score'].mean():.4f}")
print(f"Average teacher score: {df['teacher_score'].mean():.4f}")
print(f"Average reward: {df['reward'].mean():.4f}")

# By language
print("\nBy Language:")
print(df.groupby('language').agg({
    'student_score': 'mean',
    'teacher_score': 'mean',
    'reward': 'mean'
}))

# Score distribution
print("\nScore Distribution:")
print(df['student_score'].describe())
```

### Filter and Process

```python
# Filter high-quality examples
high_quality = df[df['student_score'] > 0.8]

# Filter by language
python_examples = df[df['language'] == 'python']

# Sort by reward
best_examples = df.nlargest(10, 'reward')
```

## Dataset Card

Each uploaded dataset includes an automatic dataset card with:
- Dataset description
- Data structure documentation
- Usage examples
- Training details
- Citation information

The dataset card is automatically generated and saved as `README.md` in the dataset directory.

## Best Practices

### Dataset Naming

Use descriptive names for your datasets:
- `qwen-code-rfai-dataset-v1`
- `qwen-code-rfai-python-only`
- `qwen-code-rfai-high-quality`

### Dataset Versioning

Create new versions for significant changes:
- `qwen-code-rfai-dataset-v1`
- `qwen-code-rfai-dataset-v2`

### Privacy

- Set `private: true` in config for sensitive data
- Review data before uploading
- Remove any sensitive information

### Dataset Size

- Large datasets (>1GB) may take time to upload
- Consider splitting into multiple datasets
- Use compression if needed

## Troubleshooting

### "Dataset not found"

- Check repository name is correct
- Verify you have access to the repository
- Ensure dataset was uploaded successfully

### "Upload timeout"

- Large datasets may take time
- Check internet connection
- Try uploading during off-peak hours
- Consider splitting into smaller datasets

### "Authentication failed"

- Verify `HUGGINGFACE_TOKEN` is set correctly
- Check token has write permissions
- Ensure token hasn't expired

### "Repository already exists"

- Use a different repository name
- Or delete the existing repository first
- Or use `--overwrite` flag (if supported)

## Example Workflow

1. **Train Model** (datasets collected automatically):
   ```bash
   python train_rfai.py --config config.yaml
   ```

2. **Review Datasets**:
   ```bash
   # Check dataset files
   ls -lh datasets/
   
   # View sample entries
   head -n 5 datasets/train.jsonl | jq
   ```

3. **Upload Datasets** (automatic or manual):
   ```bash
   python export_datasets.py \
       --dataset_dir ./datasets \
       --repo_id mlx-community/qwen-code-rfai-dataset
   ```

4. **Use Datasets**:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("mlx-community/qwen-code-rfai-dataset")
   ```

## References

- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [MLX Community](https://huggingface.co/mlx-community)
- [Dataset Card Guide](https://huggingface.co/docs/hub/datasets-cards)

