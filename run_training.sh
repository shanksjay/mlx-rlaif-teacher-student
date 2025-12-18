#!/bin/bash
# Convenience script to run training with proper setup

set -e

echo "=========================================="
echo "Qwen Code RFAI Training Setup"
echo "=========================================="

# Check for API key
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: Please set either OPENAI_API_KEY or ANTHROPIC_API_KEY"
    echo "  export OPENAI_API_KEY='your-key'"
    echo "  OR"
    echo "  export ANTHROPIC_API_KEY='your-key'"
    exit 1
fi

# Create necessary directories
mkdir -p data
mkdir -p checkpoints
mkdir -p logs/tensorboard

# Generate sample data if it doesn't exist
if [ ! -f "data/train.jsonl" ]; then
    echo "Generating sample training data..."
    uv run python data_utils.py
fi

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "ERROR: config.yaml not found!"
    exit 1
fi

echo ""
echo "Starting training..."
echo "To monitor progress, run in another terminal:"
echo "  uv run tensorboard --logdir ./logs/tensorboard"
echo ""
echo "=========================================="
echo ""

# Run training
uv run python train_rfai.py --config config.yaml

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: ./checkpoints"
echo "TensorBoard logs: ./logs/tensorboard"
echo "=========================================="

