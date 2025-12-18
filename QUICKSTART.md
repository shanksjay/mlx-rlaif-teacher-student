# Quick Start Guide

## Prerequisites Check

1. **Python 3.9+**: Check with `python3 --version`
2. **API Key**: Set either:
   ```bash
   export OPENAI_API_KEY="sk-..."
   # OR
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

## Installation (5 minutes)

```bash
# Install dependencies
uv pip install -r requirements.txt

# (Optional) Preload model for faster startup
uv run python preload_model.py --model Qwen/Qwen2.5-7B-Instruct

# Generate sample data
uv run python data_utils.py
```

## Run Training

### Option 1: Using the convenience script
```bash
./run_training.sh
```

### Option 2: Manual execution
```bash
# Start TensorBoard in background (optional)
uv run tensorboard --logdir ./logs/tensorboard &

# Run training
uv run python train_rfai.py --config config.yaml
```

## Monitor Training

1. **Console Output**: Watch for printed statistics:
   ```
   Step 50:
     Loss: 2.3456
     Policy Loss: 1.2345
     KL Penalty: 0.1111
     Avg Reward: 0.7890
   ```

2. **TensorBoard**: Run `uv run tensorboard --logdir ./logs/tensorboard` and open http://localhost:6006
   - View loss curves
   - Track reward progression
   - Monitor KL divergence

## Expected Output

```
2024-01-XX XX:XX:XX - INFO - Loading base model: Qwen/Qwen2.5-7B-Instruct
2024-01-XX XX:XX:XX - INFO - Using MPS (Metal Performance Shaders)
2024-01-XX XX:XX:XX - INFO - Initializing teacher model: openai/gpt-4-turbo-preview
2024-01-XX XX:XX:XX - INFO - Starting RFAI training...
2024-01-XX XX:XX:XX - INFO - Epoch 1/3
...
```

## First Run Tips

1. **Start Small**: Use `num_samples_per_prompt: 2` in config for faster testing
2. **Monitor API Costs**: Teacher API calls can be expensive
3. **Check Memory**: M5 MacBook should handle batch_size=4 with 4-bit quantization
4. **Save Checkpoints**: Checkpoints are saved every `save_steps` (default: 500)

## Troubleshooting

### "Out of Memory"
- Reduce `batch_size` to 2
- Increase `gradient_accumulation_steps` to 16
- Ensure `use_4bit: true` in config

### "API Rate Limit"
- Add delays in `TeacherModel.generate()`
- Use GPT-3.5 instead of GPT-4
- Reduce batch size

### "No module named 'transformers'"
- Run: `pip install -r requirements.txt`

## Next Steps

1. **Customize Data**: Replace `data/train.jsonl` with your code prompts
2. **Adjust Rewards**: Modify scoring weights in `TeacherModel.score_code()`
3. **Experiment**: Try different KL penalties, learning rates
4. **Validate Model**: Compare pre/post training quality:
   ```bash
   python validate_model.py --fine_tuned_path ./checkpoints/checkpoint-500
   ```
5. **Upload to Hugging Face**: Configure and upload to MLX Community (see VALIDATION_GUIDE.md)
6. **Use MLX Models**: After training, use MLX format for faster inference:
   ```bash
   python load_mlx_model.py --model_path ./checkpoints/checkpoint-500/mlx_model
   ```

## Training Time Estimates

- **Small dataset (100 samples)**: ~2-4 hours
- **Medium dataset (1000 samples)**: ~1-2 days
- **Large dataset (10000 samples)**: ~1-2 weeks

*Times vary based on API response times and model size*

