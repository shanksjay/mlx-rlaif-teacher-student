# Scripts Directory

This directory contains all Python scripts organized by functionality.

## Directory Structure

- **`training/`** - Training scripts
  - `train_rlaif.py` - Main RLAIF training script
  
- **`validation/`** - Model validation scripts
  - `validate_model.py` - Compare baseline vs fine-tuned models
  
- **`utils/`** - Utility scripts
  - `data_utils.py` - Dataset generation utilities
  - `convert_to_mlx.py` - Convert PyTorch models to MLX format
  - `preload_model.py` - Preload and cache models
  - `load_mlx_model.py` - Load and test MLX models
  - `export_datasets.py` - Export datasets to Hugging Face
  
- **`profiling/`** - Performance profiling scripts
  - `profile_model_loading.py` - Profile model loading performance
  - `profile_with_instruments.py` - Apple Instruments profiling wrapper
  
- **`visualization/`** - Visualization scripts
  - `visualize_training.py` - Training visualization utilities

## Usage

All scripts should be run from the project root:

```bash
# Training
uv run python scripts/training/train_rlaif.py --config config.yaml

# Validation
uv run python scripts/validation/validate_model.py --base_model Qwen/Qwen2.5-Coder-3B-Instruct

# Utilities
uv run python scripts/utils/data_utils.py
uv run python scripts/utils/convert_to_mlx.py --hf-path Qwen/Qwen2.5-Coder-3B-Instruct --mlx-path ./mlx_model/q8
```

## Imports

Scripts can import from each other using the `scripts` package:

```python
from scripts.training.train_rlaif import TeacherModel
from scripts.utils.data_utils import generate_sample_data
```
