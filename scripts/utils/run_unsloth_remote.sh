#!/usr/bin/env bash
set -euo pipefail

# Run the Unsloth-enabled training from your Mac via SSH on a remote NVIDIA/CUDA machine.
#
# Prereqs on the remote machine:
# - CUDA-enabled PyTorch environment
# - `uv` installed (recommended), or adapt commands to your env
# - `requirements.txt` + `requirements-cuda.txt` installed
#
# Usage:
#   scripts/utils/run_unsloth_remote.sh user@host /abs/path/on/remote /abs/path/to/config.yaml
#
# Example:
#   scripts/utils/run_unsloth_remote.sh ubuntu@1.2.3.4 /home/ubuntu/train_coding_asst /home/ubuntu/train_coding_asst/config_nvidia.yaml

REMOTE="${1:-}"
REMOTE_DIR="${2:-}"
REMOTE_CONFIG="${3:-}"

if [[ -z "${REMOTE}" || -z "${REMOTE_DIR}" || -z "${REMOTE_CONFIG}" ]]; then
  echo "Usage: $0 user@host /abs/path/on/remote /abs/path/to/config.yaml" >&2
  exit 2
fi

echo "Running Unsloth training on ${REMOTE}:${REMOTE_DIR} with config ${REMOTE_CONFIG}"
echo "Tip: ensure your config has:"
echo "  hardware.use_unsloth: true"
echo "  hardware.use_mps: false"
echo "  hardware.use_mlx_for_generation: false"
echo

ssh -t "${REMOTE}" "cd \"${REMOTE_DIR}\" && uv run python \"scripts/training/train_rlaif.py\" --config \"${REMOTE_CONFIG}\" --debug"

