# Add Epoch Health Check with Dynamic Parameter Adjustment and Enhanced Training Metrics

## Summary

This PR adds comprehensive training health monitoring and automatic hyperparameter adjustment capabilities, along with enhanced metrics tracking for better visibility into training performance. The system now automatically detects training issues and adjusts critical parameters (learning rate, KL penalty, reward weight, etc.) to optimize reward and loss trends.

## Key Features

### 1. Epoch Health Check with Dynamic Parameter Adjustment 🔧

After each epoch, the system performs a comprehensive health check that:
- **Analyzes training trends**: Monitors reward, loss, and variance across recent epochs
- **Detects issues automatically**: Identifies problems like loss spikes, reward decline, high variance, overfitting, etc.
- **Dynamically adjusts parameters**: Automatically tunes hyperparameters to optimize training:
  - **Learning Rate**: Reduced by 20% if loss is unstable/spiking
  - **KL Penalty**: Increased if loss spikes, decreased if reward declining
  - **Reward Weight**: Increased if reward not improving
  - **Max Grad Norm**: Tightened if loss spiking
  - **Reward Threshold**: Increased if variance too high
  - **Generation Temperature**: Increased if exploration needed

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

#### Generation Performance
- Added **Total Tokens Generated** tracking across all batches
- Provides visibility into total computational work done during generation

#### Backpropagation Performance  
- Added **Total Tokens Consumed** tracking across all batches
- Shows cumulative tokens processed during backpropagation

#### Teacher API Usage
- Added **Total Tokens Received** (output tokens) tracking
- Added **Input Tokens/sec** throughput metric
- Added **Output Tokens/sec** throughput metric
- Helps identify API performance bottlenecks

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

- **Bug Fix**: `training_start_time` was never being set, causing "Training Duration: Not recorded"
- Now properly tracks training start time and displays accurate duration

**Before:**
```
⏱️  Training Duration: Not recorded
```

**After:**
```
⏱️  Training Duration:
  Total Time: 2h 15m 30s (8130.5 seconds)
```

## Technical Details

### Health Check Logic

The health check analyzes multiple signals:

1. **Loss Instability Detection**:
   - Detects loss spikes (>0.2 increase)
   - Identifies loss too high (>0.5 threshold)
   - Monitors loss increasing trend

2. **Reward Performance Detection**:
   - Detects reward decreasing trend
   - Identifies reward stagnation
   - Monitors reward too low (<0.3 threshold)

3. **Variance Analysis**:
   - Detects high reward variance (>0.08)
   - Indicates inconsistent training

4. **Overfitting Detection**:
   - Identifies reward improving but loss increasing
   - Suggests potential overfitting

5. **Exploration Analysis**:
   - Detects loss improving but reward stagnant
   - Suggests need for more exploration

### Parameter Adjustment Rules

| Issue | Adjustment | Rationale |
|-------|-----------|-----------|
| Loss spiking | LR ↓ 20%, KL ↑ 20%, Grad Norm ↓ 20% | Stabilize training, prevent divergence |
| Loss too high | LR ↓ 20% | Reduce learning rate for stability |
| Reward declining | Reward Weight ↑ 15%, KL ↓ 10% | Strengthen reward signal, allow more exploration |
| Reward stagnant | Reward Weight ↑ 15% | Increase reward influence |
| High variance | Reward Threshold ↑ 0.02 | Filter more low-quality samples |
| Overfitting | KL ↑ 10% | Prevent model drift from base |
| Need exploration | Temp ↑ 10% | Increase generation diversity |

### Configuration

New config option in `config.yaml`:
```yaml
logging:
  epoch_health_check_enabled: true  # Enable dynamic parameter adjustment
```

Can be disabled if manual control is preferred.

## Benefits

1. **Automatic Optimization**: Training automatically adapts to observed trends
2. **Better Convergence**: Dynamic adjustments help stabilize loss and improve rewards
3. **Reduced Manual Tuning**: Less need for manual hyperparameter adjustment
4. **Better Visibility**: Enhanced metrics provide complete picture of training performance
5. **Early Problem Detection**: Issues are detected and addressed automatically

## Files Changed

- `scripts/training/train_rlaif.py`:
  - Added `_epoch_health_check_and_adjust()` method
  - Added total token tracking for generation and backpropagation
  - Added API throughput metrics (input/output tokens/sec)
  - Fixed training duration tracking
  - Store optimizer as instance variable for health check adjustments

- `config.yaml`:
  - Added `epoch_health_check_enabled` configuration option

- `README.md`:
  - Updated "Critical Statistics" section with new metrics
  - Added "Training Summary" documentation
  - Updated TensorBoard metrics documentation
  - Added example training summary output

## Testing

The health check system has been tested with:
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

## Future Enhancements

Potential improvements for future PRs:
- More sophisticated trend analysis (moving averages, exponential smoothing)
- Adaptive adjustment rates based on training stage
- Multi-objective optimization (balance reward vs loss vs variance)
- Early stopping based on health check signals
- Per-layer learning rate adjustments








