# Training Loop Optimizations

## Current Bottlenecks

Based on the training output, the main bottlenecks are:

1. **Generation**: Using PyTorch MPS (~0.2 tokens/sec) - should use MLX (2-3 tokens/sec)
2. **Reward Computation**: Teacher API calls (parallelized but could be optimized)
3. **Memory Operations**: Frequent cache clearing
4. **Dataset Collection**: Extending lists every batch (memory overhead)

## Optimization Strategies

### 1. Enable MLX for Generation (Critical - 5-10x speedup)

**Current**: `use_mlx_for_generation: false` in config.yaml
**Optimization**: Enable MLX if model is available

```yaml
hardware:
  use_mlx_for_generation: true  # Enable for 5-10x faster generation
  mlx_model_path: "./mlx_model"  # or "./mlx_model_q8"
  mlx_quantization: q8_bit
```

**Expected improvement**: 5-10x faster generation (0.2 → 2-3 tokens/sec)

### 2. Optimize Reward Computation

**Current**: ThreadPoolExecutor with max_workers=4
**Optimizations**:
- Batch API calls more efficiently
- Pre-fetch teacher code while generating
- Use async/await for better concurrency
- Increase worker count if API rate limits allow

### 3. Defer Dataset Collection

**Current**: Extending dataset list every batch
**Optimization**: Collect in batches, write periodically

### 4. Overlap Generation and Reward Computation

**Current**: Sequential: Generate → Rewards → Train
**Optimization**: Start reward computation while still generating

### 5. Optimize Memory Operations

**Current**: Clearing cache every batch
**Optimization**: Clear only when needed, use memory-efficient operations

### 6. Reduce Generation Tokens

**Current**: max_tokens=256
**Optimization**: Reduce to 128-192 for faster generation (if quality acceptable)

## Implementation

See the optimized training loop in `train_rfai.py` with these improvements.

