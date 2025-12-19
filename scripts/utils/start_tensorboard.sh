#!/bin/bash
# Convenience script to start TensorBoard with warnings suppressed

# Suppress pkg_resources deprecation warning
export PYTHONWARNINGS=ignore::UserWarning

# Start TensorBoard
uv run tensorboard --logdir ./logs/tensorboard "$@"
