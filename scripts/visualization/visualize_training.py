#!/usr/bin/env python3
"""
Utility script to visualize training progress from TensorBoard logs
"""

import argparse
import os
from pathlib import Path
import json


def print_training_summary(log_dir: str):
    """Print summary of training from checkpoints"""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return
    
    # Find all checkpoints
    checkpoint_dirs = sorted(
        [d for d in log_path.parent.parent.glob("checkpoints/checkpoint-*")],
        key=lambda x: int(x.name.split("-")[-1])
    )
    
    if not checkpoint_dirs:
        print("No checkpoints found")
        return
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    for checkpoint_dir in checkpoint_dirs[-3:]:  # Show last 3 checkpoints
        stats_file = checkpoint_dir / "training_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            print(f"\n{checkpoint_dir.name}:")
            print(f"  Step: {stats.get('step', 'N/A')}")
            print(f"  Epoch: {stats.get('epoch', 'N/A')}")
            print(f"  Avg Reward: {stats.get('avg_reward', 0.0):.4f}")
            print(f"  Avg Loss: {stats.get('avg_loss', 0.0):.4f}")
            print(f"  Total Samples: {stats.get('num_samples', 0)}")
    
    print("\n" + "="*80)
    print("To view detailed metrics, run:")
    print(f"  tensorboard --logdir {log_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training progress")
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs/tensorboard',
        help='Path to TensorBoard log directory'
    )
    
    args = parser.parse_args()
    print_training_summary(args.log_dir)

