# Performance Optimization Tips for MLX on Apple Silicon

## Current Performance

- **Q8 (8-bit)**: ~2.7 tokens/sec
- **Q4 (4-bit)**: Expected ~4-6 tokens/sec (2x faster)
- **Full precision**: ~1-2 tokens/sec

## Quick Performance Improvements

### 1. Use Q4 Quantization (Recommended for Speed)

Q4 quantization provides the best speed/quality tradeoff:

```bash
# Convert with Q4 quantization
uv run python convert_to_mlx.py \
    --hf-path Qwen/Qwen2.5-7B-Instruct \
    --mlx-path ./mlx_model_q4 \
    --quantize q4_bit

# Preload and test
uv run python preload_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --mlx-path ./mlx_model_q4 \
    --mlx-quantize q4_bit
```

**Expected improvement**: 2x faster than Q8 (~5-6 tokens/sec)

### 2. Optimize Generation Parameters

#### For Maximum Speed (Greedy Decoding)
```python
generated_text = mlx_generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=256,
    temp=0.0,    # Greedy (fastest)
    top_k=1,     # Only top token
    top_p=1.0,   # Disable top_p
)
```

#### For Balanced Speed/Quality
```python
generated_text = mlx_generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=256,
    temp=0.3,    # Low temperature (faster)
    top_k=20,    # Fewer candidates (faster)
    top_p=0.9,   # Slightly lower (faster)
)
```

#### For Maximum Quality
```python
generated_text = mlx_generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=256,
    temp=0.8,    # Higher temperature
    top_k=50,    # More candidates
    top_p=0.95,  # Standard top_p
)
```

### 3. Reduce Max Tokens

Shorter generations are faster:

```python
# Fast (for testing)
max_tokens=128

# Balanced
max_tokens=256

# Quality (slower)
max_tokens=512
```

### 4. Batch Processing

Process multiple prompts in parallel (if supported):

```python
# Sequential (current)
for prompt in prompts:
    generate(prompt)

# Parallel (if MLX supports it)
# Note: Check MLX documentation for batch generation support
```

### 5. Model Warmup

First generation is slower. Warm up the model:

```python
# Warmup generation
_ = mlx_generate(model, tokenizer, prompt="test", max_tokens=5)

# Now real generations will be faster
result = mlx_generate(model, tokenizer, prompt=real_prompt, max_tokens=256)
```

## Performance Comparison

| Configuration | Tokens/sec | Speedup | Quality | Memory |
|--------------|------------|--------|---------|--------|
| PyTorch MPS (full) | 0.2 | 1x | High | High |
| MLX (full precision) | 1-2 | 5-10x | High | Medium |
| MLX Q8 | 2-3 | 10-15x | High | Low |
| MLX Q4 | 4-6 | 20-30x | Good | Very Low |
| MLX Q4 + Greedy | 6-8 | 30-40x | Good | Very Low |

## Recommended Settings by Use Case

### Training (Fast Generation)
```yaml
hardware:
  use_mlx_for_generation: true
  mlx_model_path: "./mlx_model_q4"
  mlx_quantization: q4_bit

# In code:
temp=0.3
top_k=20
max_tokens=256
```

### Inference (Quality)
```yaml
hardware:
  use_mlx_for_generation: true
  mlx_model_path: "./mlx_model_q8"
  mlx_quantization: q8_bit

# In code:
temp=0.8
top_k=50
max_tokens=512
```

### Testing (Maximum Speed)
```yaml
hardware:
  use_mlx_for_generation: true
  mlx_model_path: "./mlx_model_q4"
  mlx_quantization: q4_bit

# In code:
temp=0.0  # Greedy
top_k=1
max_tokens=128
```

## Advanced Optimizations

### 1. Use Lazy Loading

Load model parameters lazily (only when needed):

```python
from mlx_lm import load
model, tokenizer = load("mlx_model", lazy=True)
```

### 2. Memory Management

Clear MLX cache between generations:

```python
import mlx.core as mx
mx.clear_cache()  # Clear MLX cache
```

### 3. Compile Model (if supported)

Some MLX versions support model compilation:

```python
# Check MLX documentation for compilation support
# This may not be available in all versions
```

### 4. Use Smaller Models

For even faster generation, consider:
- Smaller model variants (3B instead of 7B)
- Distilled models
- Specialized code models

## Benchmarking

To benchmark your setup:

```bash
# Test Q4
uv run python preload_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --mlx-path ./mlx_model_q4 \
    --mlx-quantize q4_bit

# Test Q8
uv run python preload_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --mlx-path ./mlx_model_q8 \
    --mlx-quantize q8_bit

# Compare results
```

## Troubleshooting

### Still Slow After Q4?

1. **Check model size**: Ensure Q4 model is actually smaller
2. **Check memory**: Low memory can slow down inference
3. **Check temperature**: Lower temp = faster
4. **Check max_tokens**: Shorter = faster
5. **Warm up model**: First generation is always slower

### Quality Degradation with Q4?

1. **Use Q8 instead**: Better quality, still fast
2. **Increase temperature**: temp=0.5-0.7
3. **Increase top_k**: top_k=30-50
4. **Use top_p**: top_p=0.9-0.95

## Expected Results

With Q4 quantization and optimized parameters:

- **Generation speed**: 4-6 tokens/sec (2x faster than Q8)
- **Memory usage**: ~50% less than Q8
- **Quality**: Slight degradation, still very good for code generation
- **Overall speedup**: 20-30x vs PyTorch MPS

## Next Steps

1. Convert model to Q4: `convert_to_mlx.py --quantize q4_bit`
2. Update config: `mlx_quantization: q4_bit`
3. Optimize generation params: `temp=0.3, top_k=20`
4. Benchmark and compare

