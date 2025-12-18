# Q4 Quantization Performance Optimization

## Current Performance Results

### Q8 (8-bit) Quantization
- **Speed**: ~2.7-2.8 tokens/sec
- **Memory**: ~7GB
- **Quality**: Excellent

### Q4 (4-bit) Quantization  
- **Speed**: ~2.8 tokens/sec (similar to Q8, but uses less memory)
- **Memory**: ~4GB (50% less than Q8)
- **Quality**: Very good (slight degradation)

## Why Q4 May Not Show Speed Improvement

Q4 quantization primarily reduces **memory usage**, not necessarily speed. The speed improvement comes from:
1. **Better cache utilization** (smaller model fits better in cache)
2. **Less memory bandwidth** (fewer bits to transfer)
3. **More room for other processes** (less memory pressure)

However, on Apple Silicon with unified memory, the speed difference may be minimal unless:
- You're memory-constrained
- Running multiple models simultaneously
- Using very large models (>13B parameters)

## How to Use Q4 Quantization

### Step 1: Convert Model with Q4

```bash
uv run python convert_to_mlx.py \
    --hf-path Qwen/Qwen2.5-7B-Instruct \
    --mlx-path ./mlx_model_q4 \
    --quantize q4_bit
```

**Note**: MLX quantization happens during conversion, not loading. The model weights are quantized when converted.

### Step 2: Use Q4 Model

```bash
# Preload and test
uv run python preload_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --mlx-path ./mlx_model_q4 \
    --mlx-quantize q4_bit
```

### Step 3: Update Config

```yaml
hardware:
  use_mlx_for_generation: true
  mlx_model_path: "./mlx_model_q4"
  mlx_quantization: q4_bit
```

## Additional Performance Optimizations

### 1. Reduce Max Tokens

Shorter generations = faster:

```python
# In train_rfai.py, reduce max_tokens
max_tokens=128  # Instead of 256
```

### 2. Use Greedy Decoding (Fastest)

For maximum speed, use greedy decoding (no sampling):

```python
# MLX generate with minimal parameters (greedy)
generated_text = mlx_generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=256,
    # No temperature/top_k/top_p = greedy (fastest)
)
```

### 3. Batch Processing

Process multiple prompts efficiently:

```python
# Current: Sequential
for prompt in prompts:
    generate(prompt)

# Optimized: Batch if supported
# Note: Check MLX documentation for batch generation
```

### 4. Model Warmup

First generation is slower. Warm up:

```python
# Warmup
_ = mlx_generate(model, tokenizer, prompt="test", max_tokens=5)

# Now faster
result = mlx_generate(model, tokenizer, prompt=real_prompt, max_tokens=256)
```

### 5. Reduce Number of Samples

In training config:

```yaml
num_samples_per_prompt: 2  # Instead of 4
```

## Performance Comparison

| Configuration | Tokens/sec | Memory | Quality | Use Case |
|--------------|------------|--------|---------|----------|
| PyTorch MPS | 0.2 | High | High | Baseline |
| MLX Full | 1-2 | Medium | High | Quality |
| MLX Q8 | 2.7-2.8 | Low | High | Balanced |
| MLX Q4 | 2.8 | Very Low | Very Good | Memory-constrained |

## When to Use Q4

Use Q4 when:
- ✅ Memory is limited (<16GB RAM)
- ✅ Running multiple models
- ✅ Need to fit larger models
- ✅ Quality degradation is acceptable

Use Q8 when:
- ✅ Want best quality
- ✅ Have enough memory
- ✅ Current speed is acceptable

## Expected Improvements with Q4

1. **Memory**: 50% reduction (7GB → 4GB)
2. **Speed**: Similar to Q8 (~2.8 tokens/sec)
3. **Quality**: Slight degradation (still very good)
4. **Overall**: Better for memory-constrained systems

## Troubleshooting

### Q4 Model Not Faster

This is normal! Q4 primarily reduces memory, not speed. The speed improvement comes from:
- Better cache utilization
- Less memory pressure
- Ability to run larger batches

### Quality Degradation

If quality is too low:
1. Use Q8 instead (better quality, still fast)
2. Increase model size
3. Fine-tune on quantized model

### Model Not Quantized

If the model doesn't seem quantized:
1. Check model size (Q4 should be ~4GB, Q8 ~7GB)
2. Verify conversion completed successfully
3. Check MLX version supports quantization

## Next Steps

1. **Benchmark both**: Compare Q4 vs Q8 on your system
2. **Monitor memory**: Check if Q4 helps with memory pressure
3. **Optimize parameters**: Reduce max_tokens, use greedy decoding
4. **Consider other optimizations**: Batch processing, model warmup

## Summary

Q4 quantization provides:
- ✅ 50% memory reduction
- ✅ Similar speed to Q8
- ✅ Very good quality
- ✅ Better for memory-constrained systems

For maximum speed, combine Q4 with:
- Reduced max_tokens
- Greedy decoding
- Model warmup
- Fewer samples per prompt

