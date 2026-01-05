# Controller Fix: Corrected Automatic Compensation Logic

## Critical Problem Identified

The automatic compensation system was moving parameters in the **wrong direction** when a downward reward trend was detected, making the problem worse instead of better.

### Previous (Incorrect) Behavior

When reward dropped, the controller would:
- ❌ **Decrease learning rate** by 20% → Freezes learning during instability
- ❌ **Increase reward_weight** by 25% → Amplifies noise
- ❌ **Decrease KL penalty** by 15% → Loosens constraint, increases drift risk
- ❌ **Increase generation_temperature** by 10% → Increases variance and errors

This combination was **highly counterproductive**:
- Raising temperature increases output variance and error rate (more syntax mistakes, more off-policy randomness)
- Reducing KL penalty loosens the leash while increasing reward-weighted updates → increases instability
- Increasing reward_weight amplifies noise (even with baseline logging, updates are reward-proportional and high variance)
- Decreasing LR repeatedly can freeze learning in the middle of instability

## Fixed Behavior

### Within-Epoch Trend Detection (`_adjust_config_for_downward_trend`)

When reward drops during an epoch, the controller now:
- ✅ **DECREASE temperature** (5-10% reduction) → Reduce variance and errors
- ✅ **INCREASE KL penalty** (5-15% increase) → Tighten constraint, prevent drift
- ✅ **DECREASE reward_weight** (5-15% reduction) → Reduce noise amplification
- ✅ **Slightly decrease LR** (2-5% reduction) → Minimal stabilization, not aggressive

### Hard Caps Added

To prevent dangerous values:
- **Temperature**: `TEMP_MAX = 0.9` (maximum for code generation), `TEMP_MIN = 0.5`
- **Reward Weight**: `REWARD_WEIGHT_MAX = 1.5` (until advantage normalization is fully verified)
- **KL Penalty**: `KL_PENALTY_MIN = 0.10`, `KL_PENALTY_MAX = 0.30`
- **Learning Rate**: `LR_MIN = 1e-6`

### Epoch Health Check (`_epoch_health_check_and_adjust`)

Fixed similar issues in the epoch-level health check:

1. **High Reward Variance**:
   - ✅ Now **DECREASES** temperature (was doing nothing)
   - Reduces temperature by 5% when variance > 0.08

2. **Reward Declining**:
   - ✅ **DOES NOT** increase reward_weight (was increasing by 10%)
   - ✅ **DOES NOT** reduce KL penalty (was reducing by 10%)
   - ✅ Only reduces reward_weight if it's already above the cap (1.5)
   - ✅ Keeps KL penalty stable or increases it

3. **Loss Improving but Reward Stagnant**:
   - ✅ **DOES NOT** increase temperature (was increasing by 10%)
   - ✅ Keeps temperature stable (exploration should come from `num_samples_per_prompt`, not temperature)

## Key Changes

### 1. `_adjust_config_for_downward_trend` Method

**Before**:
```python
# WRONG: Decreasing LR aggressively
lr_reduction = 0.20  # 20% reduction
new_lr = current_lr * (1.0 - lr_reduction)

# WRONG: Increasing reward weight
reward_weight_increase = 0.25  # 25% increase
new_rw = current_rw * (1.0 + reward_weight_increase)

# WRONG: Decreasing KL penalty
kl_reduction = 0.15  # 15% reduction
new_kl = current_kl * (1.0 - kl_reduction)

# WRONG: Increasing temperature
new_temp = current_temp * 1.1  # Increase by 10%
```

**After**:
```python
# CORRECT: Decrease temperature
temp_reduction = 0.10  # 10% reduction
new_temp = max(current_temp * (1.0 - temp_reduction), TEMP_MIN)
new_temp = min(new_temp, TEMP_MAX)  # Hard cap at 0.9

# CORRECT: Increase KL penalty
kl_increase = 0.15  # 15% increase
new_kl = min(current_kl * (1.0 + kl_increase), KL_PENALTY_MAX)
new_kl = max(new_kl, KL_PENALTY_MIN)  # Hard cap at 0.10

# CORRECT: Decrease reward weight
reward_weight_reduction = 0.15  # 15% reduction
new_rw = max(current_rw * (1.0 - reward_weight_reduction), 0.5)
new_rw = min(new_rw, REWARD_WEIGHT_MAX)  # Hard cap at 1.5

# CORRECT: Slightly decrease LR (not aggressive)
lr_reduction = 0.05  # 5% reduction (minimal)
new_lr = max(current_lr * (1.0 - lr_reduction), LR_MIN)
```

### 2. Epoch Health Check Fixes

**High Variance**:
```python
# Now decreases temperature when variance is high
if reward_variance > 0.08:
    new_temp = max(current_temp * 0.95, TEMP_MIN)  # Reduce by 5%
    new_temp = min(new_temp, TEMP_MAX)  # Cap at 0.9
```

**Reward Declining**:
```python
# No longer increases reward_weight or reduces KL penalty
# Only reduces reward_weight if above cap
if current_rw > REWARD_WEIGHT_MAX:
    new_rw = max(current_rw * 0.95, 1.0)  # Reduce by 5%
```

**Loss Improving but Reward Stagnant**:
```python
# No longer increases temperature
# Keep temperature stable - exploration should come from num_samples_per_prompt
```

## Expected Impact

1. **Reduced Variance**: Lower temperature → fewer syntax errors, more stable generations
2. **Better Constraint**: Higher KL penalty → prevents model drift, maintains policy stability
3. **Less Noise**: Lower reward_weight → reduces gradient variance
4. **Stable Learning**: Minimal LR reduction → doesn't freeze learning during instability

## Testing Recommendations

1. Monitor temperature: Should **never increase** when reward declines
2. Monitor KL penalty: Should **increase** or stay stable when reward declines
3. Monitor reward_weight: Should **decrease** or stay stable when reward declines (never exceed 1.5)
4. Check logs for warning messages indicating parameter adjustments
5. Verify that reward recovers faster after trend detection triggers

## Critical Rule

**NEVER increase `generation_temperature` when reward is declining or variance is high.**

Temperature increases lead to:
- More syntax errors
- Higher variance in code quality
- More off-policy randomness
- Worsening of the observed failure mode

If more exploration is needed, increase `num_samples_per_prompt` instead.






