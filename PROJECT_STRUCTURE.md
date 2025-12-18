# Project Structure

```
train_coding_asst/
├── README.md                 # Main documentation with high-level flow
├── QUICKSTART.md             # Quick start guide
├── PROJECT_STRUCTURE.md      # This file
├── requirements.txt          # Python dependencies
├── config.yaml              # Training configuration
├── .gitignore               # Git ignore rules
│
├── train_rfai.py            # Main training script with RFAI implementation
├── data_utils.py            # Dataset utilities and sample data generation
├── visualize_training.py    # Training visualization utilities
├── run_training.sh          # Convenience script to run training
│
├── data/                    # Training data directory
│   ├── .gitkeep
│   ├── train.jsonl          # Training dataset (generated)
│   └── eval.jsonl           # Evaluation dataset (generated)
│
├── checkpoints/             # Model checkpoints (created during training)
│   └── checkpoint-{step}/
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer files
│       └── training_stats.json
│
└── logs/                    # Training logs (created during training)
    └── tensorboard/
        └── events.out.tfevents.*
```

## Key Files

### `train_rfai.py`
Main training script implementing:
- RFAI (Reinforcement from AI Feedback) algorithm
- Teacher-student architecture
- Policy gradient training with KL penalty
- TensorBoard logging
- Checkpoint saving

### `config.yaml`
Configuration file for:
- Model settings (Qwen base model, quantization)
- Teacher model (OpenAI/Anthropic)
- Training hyperparameters
- RFAI parameters (rewards, KL penalty)
- Hardware settings (M5 MacBook optimizations)

### `data_utils.py`
Utilities for:
- Creating sample datasets
- Loading and validating datasets
- Data format conversion

### `visualize_training.py`
Script to:
- Print training summaries
- View checkpoint statistics
- Quick overview of training progress

## Data Format

Training data is in JSONL format:
```json
{"prompt": "Implement binary search", "language": "python"}
{"prompt": "Create thread-safe queue", "language": "cpp"}
{"prompt": "Parse JSON safely", "language": "rust"}
```

## Training Flow

1. **Load Config**: Read `config.yaml`
2. **Load Data**: Read training/eval JSONL files
3. **Initialize Models**: Load Qwen student + Teacher API
4. **Training Loop**:
   - Generate student samples
   - Get teacher scores
   - Compute rewards
   - Update model weights
   - Log metrics
5. **Save Checkpoints**: Periodically save model state
6. **Monitor**: TensorBoard visualization

## Output Files

- **Checkpoints**: Saved models at `./checkpoints/checkpoint-{step}/`
- **Logs**: TensorBoard logs at `./logs/tensorboard/`
- **Stats**: Training statistics in checkpoint directories

