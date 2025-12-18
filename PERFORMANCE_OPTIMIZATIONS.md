# Performance Optimizations for M5 MacBook

## Overview

This document describes the performance optimizations implemented for Apple M5 Silicon with 32GB unified memory.

## Model Loading Optimizations

### Fast Model Loading
- **Safetensors Format**: Uses safetensors for faster checkpoint loading (up to 2x faster)
- **Low CPU Memory Usage**: Optimizes memory during loading
- **Fast Tokenizer**: Uses fast tokenizer implementation
- **Progress Tracking**: Shows loading time for monitoring with detailed profiling
- **Memory Cache Clearing**: Clears MPS/CUDA cache before/after loading
- **dtype Parameter**: Uses `dtype` instead of deprecated `torch_dtype` (fixes warnings)
- **Memory Monitoring**: Real-time memory usage tracking during loading

### Why Last 25% is Slower

The last checkpoint shard (75-100%) typically takes longer because:

1. **Memory Pressure**: Previous shards consume memory, causing slower allocation
2. **Device Mapping**: Final shard requires completing device mapping across all layers
3. **Quantization Setup**: 4-bit quantization finalizes setup on the last shard
4. **Model Initialization**: Final model structure initialization happens at the end

**Typical timing**:
- First 3 shards (75%): ~15-20 seconds each
- Last shard (25%): ~25-35 seconds (due to overhead above)

### Preload Model (Recommended)
To avoid slow loading on every training run, preload the model once:

```bash
uv run python preload_model.py --model Qwen/Qwen2.5-7B-Instruct
```

This will:
- Download and cache the model locally
- Verify the model works correctly
- Make subsequent training runs start much faster

**First load**: ~5-10 minutes (download + load)  
**Subsequent loads**: ~1-2 minutes (from cache)

### Profile Loading

To analyze loading performance in detail:

```bash
uv run python profile_loading.py --model Qwen/Qwen2.5-7B-Instruct
```

This will show:
- Time per loading phase
- Memory usage over time
- Breakdown of where time is spent

## Key Optimizations

### 1. DataLoader Configuration
- **num_workers=0**: M5's unified memory architecture doesn't benefit from multiple workers
- **pin_memory=False**: Not needed for M5
- Eliminates tokenizer fork warnings

### 2. Batch Processing
- **Batch Student Generation**: Generate multiple samples in batches instead of sequentially
- **Reduced max_new_tokens**: 256 instead of 512 for faster generation
- **Batch tokenization**: Process all prompts together

### 3. Parallel API Calls
- **ThreadPoolExecutor**: Process teacher API calls concurrently (up to 4 workers)
- **Caching**: Cache teacher responses to avoid redundant API calls
- **Async processing**: Collect results as they complete

### 4. Reduced Sample Count
- **num_samples_per_prompt: 2** (reduced from 4)
- Significantly reduces API calls and generation time

### 5. Memory Optimizations
- **Reduced max_length**: 1024 instead of 2048
- **Smaller generation batches**: Process in chunks of 8
- **Efficient tensor operations**: Use bfloat16 throughout

### 6. Tokenizer Parallelism
- **TOKENIZERS_PARALLELISM=false**: Prevents fork warnings on M5
- Set as environment variable at module level

## Performance Improvements

### Before Optimizations
- **Time per batch**: ~5-10 minutes (with 4 samples per prompt)
- **API calls**: Sequential, blocking
- **Memory**: High usage from large batches

### After Optimizations
- **Time per batch**: ~1-2 minutes (estimated)
- **API calls**: Parallel (4 concurrent)
- **Memory**: Optimized for unified memory architecture

## Configuration Recommendations

For M5 with 32GB:

```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 8

rfai:
  num_samples_per_prompt: 2  # Reduced from 4

model:
  max_length: 1024  # Reduced from 2048

hardware:
  dataloader_num_workers: 0  # Critical for M5
```

## Monitoring Performance

Use TensorBoard to monitor:
- `System/CPU_Percent`: Should be high during training
- `System/Memory_Percent`: Monitor for memory pressure
- `System/Process_Memory_GB`: Track training process memory

## Further Optimizations

If still slow, consider:
1. **Reduce batch_size** to 2
2. **Increase gradient_accumulation_steps** to 16
3. **Use GPT-3.5** instead of GPT-4/Claude for teacher (faster API)
4. **Cache more aggressively**: Save teacher responses to disk
5. **Use local teacher model**: If available, use a local model instead of API

## Troubleshooting

### Still Slow?
- Check API response times in logs
- Monitor CPU usage (should be high)
- Check if memory is the bottleneck
- Consider reducing dataset size for testing

### Memory Issues?
- Reduce batch_size to 2
- Reduce max_length to 512
- Enable 4-bit quantization
- Close other applications

