# Quick Start: Resume Training from Epoch 4

## Simple Steps

### 1. Update config.yaml

Edit `config.yaml` and set the resume checkpoint path:

```yaml
training:
  resume_from_checkpoint: "./checkpoints/checkpoint-e4-end-gs4"
  num_epochs: 7  # Will continue from epoch 5
  # ... rest of your config
```

### 2. Run Training

```bash
uv run python scripts/training/train_rlaif.py --config config.yaml
```

That's it! The training will:
- ✅ Load the base model
- ✅ Load LoRA adapters from `checkpoint-e4-end-gs4`
- ✅ Resume from epoch 5 (next epoch after checkpoint)
- ✅ Use your new hyperparameters (lower LR, higher KL penalty, etc.)

## What Happens

1. **Model Loading**: Base model loads first, then LoRA adapters are loaded from checkpoint
2. **Epoch Resumption**: Training starts from epoch 5 (checkpoint was at epoch 4)
3. **New Hyperparameters Applied**: Your updated config values take effect:
   - Learning rate: 5e-6 (reduced from 1e-5)
   - KL penalty: 0.1 (increased from 0.05)
   - Gradient clipping: 0.5 (reduced from 1.0)
   - Warmup steps: 200 (increased from 100)

## Verify Checkpoint Exists

Before running, verify your checkpoint:

```bash
ls -la checkpoints/checkpoint-e4-end-gs4/
```

You should see:
- `adapter_model.safetensors` (LoRA weights)
- `adapter_config.json` (LoRA config)
- `training_stats.json` (epoch info)

## Expected Output

When training starts, you'll see:

```
Loading model weights...
✓ LoRA applied successfully!
Resuming from checkpoint: ./checkpoints/checkpoint-e4-end-gs4
✓ Loaded LoRA adapters from checkpoint
Will resume from epoch 5 (0-indexed: 4)
Resuming training from epoch 5 (checkpoint was at epoch 4)
Epoch 5/7
...
```

## Notes

- **Optimizer State**: Optimizer restarts from scratch (this is fine with new hyperparameters)
- **Learning Rate Schedule**: Cosine scheduler restarts (warmup will run again)
- **Epoch Counting**: Training continues from epoch 5, so you'll complete epochs 5, 6, 7

## Troubleshooting

If you see an error about checkpoint not found:
1. Check the path is correct: `./checkpoints/checkpoint-e4-end-gs4`
2. Verify the checkpoint directory exists
3. Make sure `adapter_model.safetensors` exists in the checkpoint

If training doesn't resume:
- Check the logs for "Resuming from checkpoint" message
- Verify `training_stats.json` exists in checkpoint directory









