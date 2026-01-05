# How to Resume Training from Epoch 4

## Overview

Since you're using LoRA (Low-Rank Adaptation), resuming requires loading:
1. The base model (Qwen/Qwen2.5-Coder-3B-Instruct)
2. The LoRA adapter weights from the checkpoint

## Method 1: Modify Config to Use Checkpoint (Recommended)

The training script doesn't have explicit resume functionality, but you can modify the code to load LoRA adapters from a checkpoint. Here's the easiest approach:

### Step 1: Update config.yaml

Add a `resume_from_checkpoint` field to your config:

```yaml
training:
  output_dir: "./checkpoints"
  resume_from_checkpoint: "./checkpoints/checkpoint-e4-end-gs4"  # Path to Epoch 4 checkpoint
  num_epochs: 7  # Will continue from epoch 5
  # ... rest of config
```

### Step 2: Modify train_rlaif.py to Support Resume

You'll need to add code to load LoRA adapters from checkpoint. Add this after LoRA is initialized (around line 3220):

```python
# After: self.model = get_peft_model(self.model, lora_config)

# Load LoRA adapters from checkpoint if resuming
if hasattr(config, 'resume_from_checkpoint') and config.resume_from_checkpoint:
    checkpoint_path = Path(config.resume_from_checkpoint)
    if checkpoint_path.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        from peft import PeftModel
        # Load adapter weights
        self.model = PeftModel.from_pretrained(self.model, str(checkpoint_path))
        logger.info("✓ Loaded LoRA adapters from checkpoint")
        
        # Load training stats to resume epoch counting
        stats_file = checkpoint_path / "training_stats.json"
        if stats_file.exists():
            import json
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            resume_epoch = stats.get('epoch', 0)
            logger.info(f"Resuming from epoch {resume_epoch + 1}")
```

## Method 2: Quick Code Modification (Simpler)

### Step 1: Find where LoRA is initialized

In `scripts/training/train_rlaif.py`, find the section around line 3220 where LoRA is set up:

```python
self.model = get_peft_model(self.model, lora_config)
```

### Step 2: Add checkpoint loading right after

Add this code immediately after the `get_peft_model` line:

```python
# Resume from checkpoint if specified
checkpoint_path = "./checkpoints/checkpoint-e4-end-gs4"  # Change this to your checkpoint
if Path(checkpoint_path).exists():
    logger.info(f"Loading LoRA adapters from checkpoint: {checkpoint_path}")
    from peft import PeftModel
    self.model = PeftModel.from_pretrained(self.model, str(checkpoint_path))
    logger.info("✓ Resumed from checkpoint")
```

### Step 3: Update epoch loop to start from epoch 5

Find the epoch loop (around line 4112):

```python
for epoch in range(self.config.num_epochs):
```

Change it to:

```python
start_epoch = 4  # Start from epoch 5 (0-indexed, so 4 = epoch 5)
for epoch in range(start_epoch, self.config.num_epochs):
```

## Method 3: Use Checkpoint as Base Model (Advanced)

If you want to merge the LoRA adapters into the base model first:

### Step 1: Merge LoRA adapters

Create a script to merge:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    torch_dtype=torch.bfloat16
)

# Load LoRA adapters
peft_model = PeftModel.from_pretrained(base_model, "./checkpoints/checkpoint-e4-end-gs4")

# Merge adapters into base model
merged_model = peft_model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./checkpoints/merged-e4")
```

### Step 2: Use merged model as base

In `config.yaml`:

```yaml
model:
  base_model: "./checkpoints/merged-e4"
```

Then disable LoRA:

```yaml
rlaif:
  use_lora: false
```

**Note**: This approach loses the ability to continue LoRA training, so it's not recommended if you want to keep using LoRA.

## Recommended Approach: Method 2 (Quick Code Modification)

For the fastest solution, use Method 2:

1. **Edit `scripts/training/train_rlaif.py`**:
   - After line ~3220 (where `get_peft_model` is called), add the checkpoint loading code
   - Change the epoch loop to start from epoch 4 (which means epoch 5 in 1-indexed)

2. **Run training**:
   ```bash
   uv run python scripts/training/train_rlaif.py --config config.yaml
   ```

3. **Training will**:
   - Load base model
   - Load LoRA adapters from `checkpoint-e4-end-gs4`
   - Continue training from epoch 5 with your new hyperparameters (lower LR, higher KL penalty, etc.)

## Verify Checkpoint Contents

Before resuming, verify your checkpoint has the necessary files:

```bash
ls -la checkpoints/checkpoint-e4-end-gs4/
```

You should see:
- `adapter_model.safetensors` (LoRA weights)
- `adapter_config.json` (LoRA configuration)
- `training_stats.json` (training state)
- `checkpoint_summary.json` (epoch info)

## Important Notes

1. **Optimizer State**: The current implementation doesn't save/load optimizer state, so training will restart the optimizer from scratch. This is usually fine with the new hyperparameters.

2. **Learning Rate Schedule**: The cosine scheduler will restart from the beginning. If you want to continue the schedule, you'll need to adjust `warmup_steps` or modify the scheduler initialization.

3. **Epoch Counting**: Make sure to adjust the epoch loop to start from epoch 5 (index 4) to avoid re-training epochs 1-4.

4. **New Hyperparameters**: Your updated config (lower LR, higher KL penalty) will be applied from epoch 5 onwards, which should help stabilize training.

## Expected Behavior After Resume

- Training will start from epoch 5
- Loss should stabilize (target: 0.2-0.3 range)
- Reward should improve consistently (no regression)
- Best-of-N reward should continue improving
- Training will use your new hyperparameters:
  - Learning rate: 5e-6 (reduced from 1e-5)
  - KL penalty: 0.1 (increased from 0.05)
  - Gradient clipping: 0.5 (reduced from 1.0)
  - Warmup steps: 200 (increased from 100)









