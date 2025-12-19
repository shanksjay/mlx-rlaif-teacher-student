# TensorBoard Metrics Reference

This document lists all metrics logged to TensorBoard during training.

## Training Metrics (`Train/`)

- **`Train/Loss`**: Total training loss (policy loss + KL penalty)
- **`Train/PolicyLoss`**: Policy gradient loss
- **`Train/KLPenalty`**: KL divergence penalty
- **`Train/AvgReward`**: Average reward for the batch
- **`Train/RewardStd`**: Standard deviation of rewards (consistency measure)
- **`Train/RewardMin`**: Minimum reward in batch
- **`Train/RewardMax`**: Maximum reward in batch
- **`Train/RewardVariance`**: Variance of rewards (lower = more consistent)

## Epoch Metrics (`Epoch/`)

- **`Epoch/AvgReward`**: Average reward across the epoch
- **`Epoch/AvgLoss`**: Average loss across the epoch
- **`Epoch/RewardVariance`**: Reward variance across the epoch (lower = more consistent)
- **`Epoch/RewardTrend`**: Change in reward from previous epoch (positive = improving)
- **`Epoch/LossTrend`**: Change in loss from previous epoch (positive = improving, as loss should decrease)
- **`Epoch/APITokens`**: Total API tokens sent to teacher model this epoch
- **`Epoch/NumSamples`**: Number of training samples processed this epoch

## Performance Metrics (`Performance/`)

### Generation Performance
- **`Performance/Generation_TokensPerSec`**: Real-time generation speed (logged every 10 batches)
- **`Performance/Generation_Time`**: Time taken for generation (logged every 10 batches)
- **`Performance/Generation_AvgTokensPerSec`**: Average generation speed across all epochs
- **`Performance/Generation_P99TokensPerSec`**: P99 generation speed (99th percentile)

### Backpropagation Performance
- **`Performance/Backprop_TokensPerSec`**: Real-time backpropagation speed (logged at logging steps)
- **`Performance/Backprop_AvgTokensPerSec`**: Average backpropagation speed across all epochs
- **`Performance/Backprop_P99TokensPerSec`**: P99 backpropagation speed (99th percentile)

## Scoring Metrics (`Scoring/`)

These represent the scoring weights used for evaluation:
- **`Scoring/Correctness`**: Correctness weight (0.3)
- **`Scoring/CodeQuality`**: Code quality weight (0.3)
- **`Scoring/Efficiency`**: Efficiency weight (0.2)
- **`Scoring/Documentation`**: Documentation weight (0.2)

*Note: Currently these are fixed weights. Future versions may track actual criterion scores.*

## System Metrics (`System/`)

### CPU and Memory
- **`System/CPU_Percent`**: Overall CPU usage percentage
- **`System/Memory_Percent`**: System memory usage percentage
- **`System/Memory_Used_GB`**: System memory used (GB)
- **`System/Memory_Available_GB`**: System memory available (GB)
- **`System/Process_Memory_GB`**: Process memory usage (RSS, GB)

### GPU/MPS Metrics
- **`System/GPU_Memory_Used_GB`**: GPU/MPS memory used (GB)
- **`System/GPU_Memory_Total_GB`**: Total GPU/MPS memory (GB)
- **`System/GPU_Memory_Percent`**: GPU/MPS memory usage percentage
- **`System/GPU_Utilization`**: Estimated GPU/Neural Engine utilization (%)

## Viewing Metrics in TensorBoard

1. Start TensorBoard:
   ```bash
   # Suppress pkg_resources deprecation warning
   PYTHONWARNINGS=ignore::UserWarning uv run tensorboard --logdir ./logs/tensorboard
   ```

2. Open in browser:
   - Navigate to `http://localhost:6006`
   - Select metrics from the left sidebar
   - Use the time range selector to focus on specific epochs

## Key Metrics to Monitor

### Training Health
- **`Epoch/AvgReward`**: Should increase over epochs (target: 0.7+)
- **`Epoch/AvgLoss`**: Should decrease and stabilize
- **`Epoch/RewardVariance`**: Should be low (< 0.01) for consistent training

### Performance Bottlenecks
- **`Performance/Generation_TokensPerSec`**: If < 1.0, consider enabling MLX
- **`Performance/Backprop_TokensPerSec`**: Monitor for training efficiency

### System Resources
- **`System/GPU_Utilization`**: Should be > 50% for efficient GPU usage
- **`System/Memory_Percent`**: Monitor for OOM warnings (> 90%)

### Convergence Indicators
- **`Epoch/RewardTrend`**: Should be positive and decreasing (approaching 0)
- **`Epoch/LossTrend`**: Should be positive (loss decreasing) and stabilizing

## Metric Groups

### Reward Analysis
- `Epoch/AvgReward`
- `Epoch/RewardTrend`
- `Epoch/RewardVariance`
- `Train/RewardStd`
- `Train/RewardVariance`

### Loss Analysis
- `Epoch/AvgLoss`
- `Epoch/LossTrend`
- `Train/Loss`
- `Train/PolicyLoss`
- `Train/KLPenalty`

### Performance Analysis
- `Performance/Generation_AvgTokensPerSec`
- `Performance/Generation_P99TokensPerSec`
- `Performance/Backprop_AvgTokensPerSec`
- `Performance/Backprop_P99TokensPerSec`

### System Health
- `System/CPU_Percent`
- `System/Memory_Percent`
- `System/GPU_Utilization`
- `System/GPU_Memory_Percent`
