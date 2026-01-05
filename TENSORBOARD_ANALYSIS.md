# TensorBoard Charts Correlation Analysis

## Executive Summary

The TensorBoard charts reveal **cyclical training patterns** that correlate with:
1. **Epoch boundaries** (~60-70 steps per cycle)
2. **Student score cache clearing** at epoch start
3. **Memory fragmentation** causing generation performance drops
4. **Training instability** from high loss volatility

---

## Chart Analysis

### 1. **Reward_EMA Cyclical Pattern** (Every ~60-70 Steps)

**Observation:**
- Reward EMA starts high (~0.4), gradually decreases to low (~0.24-0.26), then rises again
- Pattern repeats approximately 3 times over 180 steps
- Each cycle lasts ~60-70 steps

**Root Cause:**
```python
# Line 4446-4455: Student score cache cleared at epoch start
# CRITICAL: Clear student score cache at the start of each epoch
# This ensures the model gets fresh feedback even if it generates similar code
keys_to_remove = [k for k in self.teacher_score_cache.keys() if not k.startswith("TEACHER_CODE:")]
for key in keys_to_remove:
    del self.teacher_score_cache[key]
```

**What's Happening:**
1. **Early in Epoch**: Model generates code → gets cached scores → reward appears stable/high
2. **Mid Epoch**: Model improves, generates better code → but similar code gets re-scored → reward fluctuates
3. **End of Epoch**: Cache is full, some code gets fresh scores → reward variance increases
4. **Epoch Boundary**: Cache cleared → model generates similar code but gets **fresh scores** → reward resets/fluctuates

**Impact on Generation Performance:**
- Early epoch: High cache hit rate → fast scoring → good generation throughput
- Mid epoch: Cache misses increase → slower scoring → generation appears slower
- Cache clear: Fresh API calls → temporary slowdown → then recovery

---

### 2. **Reward_GainFromBaseline Cyclical Pattern**

**Observation:**
- Mirrors Reward_EMA pattern exactly
- Oscillates between +0.06 and -0.06
- 3 full cycles over 180 steps

**Correlation:**
- When Reward_EMA is high → Gain is positive
- When Reward_EMA is low → Gain is negative
- This indicates the model is **learning** (improving from baseline) but the **cache clearing disrupts the measurement**

**Key Insight:**
The model IS improving, but the cyclical cache clearing makes it appear unstable.

---

### 3. **Loss_Policy High Volatility** (0.45-0.85 range)

**Observation:**
- Extreme fluctuations with no clear trend
- Values spike between 0.5-0.95 frequently
- No convergence visible

**Root Causes:**
1. **High learning rate** (even after reductions)
2. **Reward weight changes** causing policy gradient instability
3. **KL penalty adjustments** affecting loss balance
4. **Small batch size** → high variance in gradient estimates

**Impact:**
- Unstable gradients → unstable parameter updates → generation performance degrades
- Model "forgets" good patterns → needs to relearn → wastes generation time

---

### 4. **Loss_KL Volatility** (-0.002 to 0.01 range)

**Observation:**
- Much smaller scale than policy loss
- Still highly oscillatory
- Fluctuates around zero

**What This Means:**
- KL penalty is working (keeping model close to reference)
- But the **magnitude of fluctuations** suggests the model is oscillating between:
  - High reward (but far from reference) → high KL penalty
  - Low reward (close to reference) → low KL penalty

**Correlation with Generation:**
- When KL is high → model is exploring → more diverse generation → slower (more API calls)
- When KL is low → model is conservative → faster generation → but lower quality

---

## Correlation with Generation Tokens/Sec Drops

### Pattern Identified:

1. **Early Epoch Batches (Steps 0-15):**
   - High generation tokens/sec (~48 tok/s)
   - **Why**: Clean memory state, high cache hit rate, stable model state

2. **Mid Epoch (Steps 15-55):**
   - Performance drops to ~29-30 tok/s
   - **Why**: 
     - Memory fragmentation builds up
     - Cache misses increase (student code cache filling)
     - Loss volatility causes model instability
     - More diverse generation → slower decoding

3. **End of Epoch (Steps 55-65):**
   - Sharp performance spike (~48 tok/s)
   - **Why**: 
     - Cache clear event (fragmentation health check)
     - Memory freed → temporary performance boost
     - Then immediate drop as fragmentation rebuilds

4. **Next Epoch Start:**
   - Performance resets high
   - **Why**: Student score cache cleared → fresh start

---

## Root Cause Chain

```
Epoch Start
  ↓
Student Score Cache Cleared (Line 4446-4455)
  ↓
Model Generates Code → Gets Fresh Scores
  ↓
Reward EMA Resets/Adjusts (but EMA continues across epochs)
  ↓
Early Epoch: High Cache Hits → Fast Scoring → Good Generation
  ↓
Mid Epoch: Cache Fills → More Misses → Slower Scoring
  ↓
Memory Fragmentation Builds (MPS/Metal)
  ↓
Generation Performance Drops (29-30 tok/s)
  ↓
Loss Volatility Increases (unstable training)
  ↓
Fragmentation Health Check Triggers Cache Clear
  ↓
Temporary Performance Spike
  ↓
Cycle Repeats at Next Epoch
```

---

## Recommendations

### 1. **Fix Reward Measurement Instability**

**Problem:** Cache clearing causes reward to appear cyclical even when model is improving

**Solution Options:**
- **Option A**: Don't clear student cache at epoch boundaries (only clear when truly needed)
- **Option B**: Track "fresh scores" vs "cached scores" separately in metrics
- **Option C**: Use a separate validation set for epoch-level reward tracking

**Code Change:**
```python
# Instead of clearing all student scores, use a smarter strategy:
# - Keep scores for code that hasn't changed significantly
# - Only clear scores for code that the model has likely improved
```

### 2. **Reduce Loss Volatility**

**Problem:** High loss volatility causes training instability

**Solution:**
- Reduce learning rate more aggressively when loss is volatile
- Increase gradient accumulation steps to reduce variance
- Use gradient clipping more conservatively
- Consider increasing batch size if memory allows

**Current Health Check:** Already adjusts LR, but may need more aggressive reduction for volatility

### 3. **Optimize Cache Management**

**Problem:** Cache clearing causes performance spikes and drops

**Solution:**
- Implement incremental cache eviction (LRU) instead of full clears
- Set cache size limits to prevent unbounded growth
- Clear cache based on age/usage, not just at epoch boundaries

### 4. **Improve Memory Management**

**Problem:** Memory fragmentation causes generation slowdown

**Solution:**
- More frequent but smaller cache clears (proactive, not reactive)
- Monitor fragmentation continuously, not just at health checks
- Consider using MLX cache limits more aggressively

### 5. **Stabilize Training**

**Problem:** Reward weight and KL penalty changes cause instability

**Solution:**
- Make health check adjustments more conservative
- Use exponential moving averages for hyperparameter adjustments
- Add "cooldown" periods after adjustments

---

## Expected Improvements After Fixes

1. **Reward_EMA**: Should show steady upward trend instead of cycles
2. **Loss_Policy**: Should show decreasing trend with less volatility
3. **Generation Tokens/Sec**: Should remain more stable throughout epoch
4. **Reward_GainFromBaseline**: Should show consistent positive trend

---

## Monitoring Recommendations

Add these TensorBoard metrics to track the correlations:

1. **Cache Hit Rate by Epoch Phase**: Track how cache hits change within epoch
2. **Memory Fragmentation Over Time**: Correlate with generation performance
3. **Cache Clear Events**: Mark when cache clears happen (already added)
4. **Reward Fresh vs Cached**: Separate metrics for fresh vs cached scores
5. **Generation Performance by Cache State**: Track tok/s vs cache hit rate

---

## Conclusion

The cyclical patterns are **artifacts of the training process**, not actual model behavior:
- **Reward cycles** = Cache clearing at epoch boundaries
- **Loss volatility** = Training instability from hyperparameter adjustments
- **Generation drops** = Memory fragmentation + cache management

Fixing these will reveal the **true training progress** and improve generation performance stability.







