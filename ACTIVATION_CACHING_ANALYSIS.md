# Activation Caching Analysis

## Current Implementation

The training step performs two forward passes:
1. **Reference forward pass** (eval mode, no_grad): Computes reference logits for KL divergence
2. **Training forward pass** (train mode, with gradients): Computes training logits for policy gradient

## Memory Cost Analysis

### Model Configuration
- **Model**: Qwen2.5-Coder-3B-Instruct (3B parameters)
- **Hidden size**: 3,584
- **Number of layers**: 28
- **Vocab size**: 152,064
- **Max sequence length**: 512
- **Batch size**: 2
- **Data type**: bfloat16 (2 bytes per element)

### Activation Memory Per Layer (bfloat16)

For a transformer layer with:
- **Attention activations**:
  - Query/Key/Value projections: `[batch, seq_len, hidden_size]` = `[2, 512, 3584]` = 3.67 MB per tensor × 3 = 11 MB
  - Attention scores: `[batch, heads, seq_len, seq_len]` = `[2, 28, 512, 512]` = 14.7 MB
  - Attention output: `[batch, seq_len, hidden_size]` = 3.67 MB
  - **Total attention**: ~30 MB per layer

- **MLP activations**:
  - Gate/Up projections: `[batch, seq_len, intermediate_size]` = `[2, 512, 18944]` = 19.4 MB per tensor × 2 = 38.8 MB
  - Down projection: `[batch, seq_len, hidden_size]` = 3.67 MB
  - **Total MLP**: ~42 MB per layer

- **Layer norm activations**: ~7 MB per layer

**Total per layer**: ~79 MB
**Total for 28 layers**: ~2.2 GB

### Additional Activations
- **Embeddings**: `[batch, seq_len, hidden_size]` = 3.67 MB
- **Final logits**: `[batch, seq_len, vocab_size]` = `[2, 512, 152064]` = 155.6 MB

### Total Activation Memory
- **Per forward pass**: ~2.4 GB
- **Caching reference activations**: +2.4 GB
- **Total with caching**: ~4.8 GB (just for activations)

## Optimization Strategy

### Option 1: Cache Only Logits (Current - Minimal Memory)
**Memory cost**: ~156 MB (just logits)
**Benefit**: Already implemented - we cache `ref_selected_log_probs`
**Status**: ✅ Already optimal for memory

### Option 2: Cache Intermediate Activations (High Memory)
**Memory cost**: ~2.4 GB
**Benefit**: Could potentially reuse base model activations, but:
- Eval mode vs train mode produce different activations (dropout, batch norm)
- LoRA adapters need to be applied during training pass anyway
- **Not recommended** - memory cost too high for minimal benefit

### Option 3: Single Forward Pass with Dual Output (Best Performance)
**Memory cost**: ~2.4 GB (same as current)
**Benefit**: 
- Run model once in train mode
- Compute both reference and training logits in same pass
- Use a frozen copy of base model weights for reference computation
- **Challenge**: Need separate base model or clever weight sharing

### Option 4: Hybrid Approach (Recommended)
**Memory cost**: ~156 MB (logits only)
**Implementation**:
1. Keep reference pass in eval mode (needed for proper KL divergence)
2. Cache only the final logits (already done)
3. Use gradient checkpointing to reduce memory during training pass
4. **Result**: No additional memory, but we still do 2 forward passes

## Recommendation

**Current implementation is already optimal** for memory-constrained scenarios. The two forward passes are necessary because:
1. Reference pass must be in eval mode (for proper KL divergence)
2. Training pass must be in train mode (for proper gradients)
3. These produce different activations, so caching intermediate activations won't help

**Potential optimization**: If memory allows, we could:
- Use a separate frozen base model for reference (true RLAIF setup)
- This would allow running both passes in parallel or caching base model activations
- But this doubles model memory (3B → 6B parameters in memory)

## Conclusion

For a 3B model with batch_size=2:
- **Current memory**: ~2.4 GB per forward pass
- **Caching activations**: +2.4 GB = 4.8 GB total
- **Benefit**: Minimal (can't reuse due to eval/train mode differences)
- **Recommendation**: Keep current implementation, cache only logits (already done)

The current implementation already caches `ref_selected_log_probs`, which is the optimal balance between memory and performance.






