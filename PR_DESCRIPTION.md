# Add Epoch Health Check with Dynamic Parameter Adjustment and Enhanced Training Metrics

## Summary

This PR adds comprehensive training health monitoring and automatic hyperparameter adjustment capabilities, along with enhanced metrics tracking for better visibility into training performance. The system now automatically detects training issues and adjusts critical parameters (learning rate, KL penalty, reward weight, etc.) to optimize reward and loss trends.

## Changes

### 1. Epoch Health Check with Dynamic Parameter Adjustment 🔧

**File: `scripts/training/train_rlaif.py`**

Added `_epoch_health_check_and_adjust()` method that:
- Analyzes training trends after each epoch (reward, loss, variance)
- Detects issues automatically (loss spikes, reward decline, high variance, overfitting)
- Dynamically adjusts hyperparameters:
  - **Learning Rate**: Reduced by 20% if loss is unstable/spiking
  - **KL Penalty**: Increased if loss spikes, decreased if reward declining
  - **Reward Weight**: Increased if reward not improving
  - **Max Grad Norm**: Tightened if loss spiking
  - **Reward Threshold**: Increased if variance too high
  - **Generation Temperature**: Increased if exploration needed
- Logs all adjustments to TensorBoard under `HealthCheck/` metrics
- Stores optimizer as instance variable for real-time LR updates

**Example Output:**
```
================================================================================
🔍 EPOCH 2 HEALTH CHECK
================================================================================
⚠️  Detected Issues: Loss increasing/too high, High reward variance
  Current Reward: 0.2485 (trend: +0.0087)
  Current Loss: 0.8188 (trend: -0.3626)
  Reward Variance: 0.0748

🔧 Parameter Adjustments:
  learning_rate: 0.000004 → 0.000003 (-20.0%)
  kl_penalty: 0.150000 → 0.180000 (+20.0%)
  max_grad_norm: 0.300000 → 0.240000 (-20.0%)
  reward_threshold: 0.250000 → 0.270000 (+8.0%)
================================================================================
```

### 2. Enhanced Training Metrics 📊

**File: `scripts/training/train_rlaif.py`**

#### Generation Performance
- Added `generation_tokens_total` tracking (cumulative tokens generated)
- Displays "Total Tokens Generated" in training summary

#### Backpropagation Performance
- Added `backprop_tokens_total` tracking (cumulative tokens processed)
- Displays "Total Tokens Consumed" in training summary

#### Teacher API Usage
- Added `api_time_total` tracking (total API call time)
- Added **Input Tokens/sec** calculation and display
- Added **Output Tokens/sec** calculation and display
- Displays "Total Tokens Received" (output tokens)

**Example Output:**
```
📊 Generation Performance:
  Average: 31.43 tokens/sec
  P99:     58.37 tokens/sec
  Samples:  350
  Total Tokens Generated: 76,505

🔄 Backpropagation Performance:
  Average: 782.01 tokens/sec
  P99:     822.40 tokens/sec
  Samples: 347
  Total Tokens Consumed: 51,200

🌐 Teacher API Usage:
  Total Tokens Sent: 3,813,261
  Total Tokens Received: 4,991
  Input Tokens/sec: 1,234.56
  Output Tokens/sec: 1.62
  Average per Epoch: 540,230
```

### 3. Fixed Training Duration Tracking 🐛

**File: `scripts/training/train_rlaif.py`**

- **Bug Fix**: `training_start_time` was never being set in `train()` method
- Now properly records start time at beginning of training
- Training duration now displays correctly in summary

**Before:**
```
⏱️  Training Duration: Not recorded
```

**After:**
```
⏱️  Training Duration:
  Total Time: 2h 15m 30s (8130.5 seconds)
```

### 4. Configuration Updates

**File: `config.yaml`**

- Added `logging.epoch_health_check_enabled: true` option
- Allows disabling health check if manual control is preferred

### 5. Documentation Updates

**File: `README.md`**

- Updated "Critical Statistics" section with new training summary metrics
- Added "Training Summary" subsection documenting all new metrics
- Updated TensorBoard metrics section with cumulative token tracking
- Added example training summary output in "Monitor Training" section

## Technical Details

### Health Check Detection Rules

| Issue | Detection Criteria | Adjustment |
|-------|-------------------|------------|
| Loss spiking | Loss increase >0.2 between epochs | LR ↓ 20%, KL ↑ 20%, Grad Norm ↓ 20% |
| Loss too high | Loss >0.5 | LR ↓ 20% |
| Reward declining | Reward decreasing trend | Reward Weight ↑ 15%, KL ↓ 10% |
| Reward stagnant | Reward change <0.01 over 3 epochs | Reward Weight ↑ 15% |
| High variance | Variance >0.08 | Reward Threshold ↑ 0.02 |
| Overfitting | Reward ↑ but Loss ↑ | KL ↑ 10% |
| Need exploration | Loss ↓ but Reward stagnant | Temp ↑ 10% |

### Parameter Adjustment Bounds

- Learning Rate: Minimum 1e-7
- KL Penalty: Range 0.05 - 0.3
- Reward Weight: Maximum 3.0
- Max Grad Norm: Minimum 0.1
- Reward Threshold: Maximum 0.3
- Generation Temperature: Maximum 1.5

## Benefits

1. **Automatic Optimization**: Training adapts to observed trends without manual intervention
2. **Better Convergence**: Dynamic adjustments stabilize loss and improve rewards
3. **Reduced Manual Tuning**: Less need for manual hyperparameter adjustment
4. **Better Visibility**: Enhanced metrics provide complete picture of training performance
5. **Early Problem Detection**: Issues detected and addressed automatically

## Files Changed

- `scripts/training/train_rlaif.py`: 
  - Added `_epoch_health_check_and_adjust()` method (~200 lines)
  - Added token tracking fields to `training_metrics`
  - Added token accumulation in generation and backprop paths
  - Added API time tracking and throughput calculations
  - Fixed `training_start_time` initialization
  - Store optimizer as instance variable
  - Updated `_print_training_summary()` with new metrics

- `config.yaml`:
  - Added `epoch_health_check_enabled` configuration option

- `README.md`:
  - Updated "Critical Statistics" section
  - Added "Training Summary" documentation
  - Updated TensorBoard metrics documentation
  - Added example output

## Testing

The health check system has been validated with:
- Loss spike scenarios (automatically reduces LR and increases KL)
- Reward stagnation (automatically increases reward weight)
- High variance (automatically increases reward threshold)
- Overfitting detection (automatically increases KL penalty)

All parameter adjustments are logged to TensorBoard under `HealthCheck/` metrics for monitoring.

## Backward Compatibility

- All changes are backward compatible
- Health check is enabled by default but can be disabled via config
- Existing training runs continue to work without changes
- New metrics are additive (don't break existing functionality)
