#!/usr/bin/env bash
set -euo pipefail

# `--test_prompts` expects a JSONL file path (not a raw prompt string).
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set. Export it in your shell (do not hardcode it in scripts)." >&2
  exit 1
fi

uv run python scripts/validation/validate_checkpoints.py \
  --checkpoints_dir ./checkpoints \
  --baseline_mlx_path ./mlx_model/q4 \
  --teacher_provider anthropic \
  --test_prompts ./data/eval.jsonl \
  --max_samples 50 \
  --generation_temperature 1.0 \
  --generation_seed 0 \
  --num_generations_per_prompt 6 \
  --aggregate mean \
  --score_mode both \
  --selection_metric avg_fine_tuned_reward_normalized \
  --output ./validation_checkpoints_summary.json
