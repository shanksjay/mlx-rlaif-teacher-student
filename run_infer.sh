# Step 1: Convert checkpoint to MLX (with MPS for faster merging on Apple Silicon)
uv run python scripts/utils/convert_lora_to_mlx.py \
    --checkpoint_path ./checkpoints/checkpoint-e1-end-gs20 \
    --base_model Qwen/Qwen2.5-Coder-3B-Instruct \
    --mlx_path ./checkpoints/checkpoint-e1-end-gs20/mlx_model \
    --quantize q4_bit \
    --device mps


# Step 2: Generate code with language-specific settings and compare with baseline model
uv run python scripts/inference/generate_code.py \
    --model_path ./checkpoints/checkpoint-e1-end-gs20/mlx_model \
    --baseline_model_path ./mlx_model/q4 \
    --prompt "Implement a lock-free queue in C++ using atomic operations" \
    --language cpp \
    --temperature 0.0 \
    --top_p 0.9 \
    --top_k 50 \
    --max_tokens 512 \
    --use_thinking
