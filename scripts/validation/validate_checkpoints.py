#!/usr/bin/env python3
"""
Validate multiple checkpoints and select the best one.

Typical usage (Apple Silicon / MLX):

  uv run python scripts/validation/validate_checkpoints.py \
    --checkpoints_dir ./checkpoints \
    --baseline_mlx_path ./mlx_model/q4 \
    --teacher_provider anthropic \
    --test_prompts ./data/eval.jsonl \
    --max_samples 50 \
    --generation_temperature 0.2 \
    --num_generations_per_prompt 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path so `import scripts.*` works when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.validation.validate_model import ModelValidator, load_test_prompts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointResult:
    checkpoint: str
    avg_baseline_score: float
    avg_fine_tuned_score: float
    avg_improvement: float
    num_test_cases: int


def _is_checkpoint_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    # HF checkpoint dirs usually have config.json, and MLX exports have config.json too.
    # We accept anything that looks like our training checkpoints: "checkpoint-*/"
    return p.name.startswith("checkpoint-")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate multiple checkpoints and select best by avg_fine_tuned_score")
    ap.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Directory containing checkpoint-* subdirs")
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--baseline_mlx_path", type=str, default="./mlx_model/q4")
    ap.add_argument("--teacher_provider", type=str, default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--teacher_model", type=str, default=None)
    ap.add_argument(
        "--allow_no_teacher",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, allow running without a teacher API key (falls back to heuristic scoring).",
    )
    ap.add_argument("--test_prompts", type=str, default="./data/eval.jsonl")
    ap.add_argument("--max_samples", type=int, default=50, help="0 = all prompts")
    ap.add_argument("--use_mlx", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--generation_temperature", type=float, default=0.2)
    ap.add_argument("--generation_top_p", type=float, default=0.95)
    ap.add_argument("--generation_top_k", type=int, default=50)
    ap.add_argument("--generation_seed", type=int, default=0, help="-1 disables seeding")
    ap.add_argument("--num_generations_per_prompt", type=int, default=1)
    ap.add_argument("--aggregate", type=str, default="mean", choices=["mean", "max"])
    ap.add_argument(
        "--score_mode",
        type=str,
        default="both",
        choices=["absolute", "normalized", "both"],
        help="Scoring mode (see validate_model.py). Default 'both' so we can select by normalized reward.",
    )
    ap.add_argument(
        "--selection_metric",
        type=str,
        default="avg_fine_tuned_reward_normalized",
        choices=[
            "avg_fine_tuned_score",
            "avg_fine_tuned_reward_normalized",
            "avg_improvement",
            "avg_reward_improvement",
        ],
        help="Metric to pick the best checkpoint. Default matches training reward scale.",
    )
    ap.add_argument("--output", type=str, default="./validation_checkpoints_summary.json")
    args = ap.parse_args()

    ckpt_root = Path(args.checkpoints_dir)
    ckpts = sorted([p for p in ckpt_root.iterdir() if _is_checkpoint_dir(p)], key=lambda p: p.name)
    if not ckpts:
        raise SystemExit(f"No checkpoint-* dirs found in: {ckpt_root}")

    prompts = load_test_prompts(args.test_prompts)
    if int(args.max_samples) > 0:
        prompts = prompts[: int(args.max_samples)]
    logger.info(f"Loaded {len(prompts)} prompts from {args.test_prompts}")

    results: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_metric_val: float | None = None

    for p in ckpts:
        logger.info(f"Validating checkpoint: {p}")
        validator = ModelValidator(
            base_model=args.base_model,
            fine_tuned_path=str(p),
            teacher_provider=args.teacher_provider,
            teacher_model=args.teacher_model,
            use_mlx=bool(args.use_mlx),
            baseline_mlx_path=args.baseline_mlx_path,
            fine_tuned_mlx_path=None,
            print_chars=0,
            generation_temperature=float(args.generation_temperature),
            generation_top_p=float(args.generation_top_p),
            generation_top_k=int(args.generation_top_k),
            generation_seed=(None if int(args.generation_seed) < 0 else int(args.generation_seed)),
            allow_no_teacher=bool(args.allow_no_teacher),
            score_mode=str(args.score_mode),
        )
        r = validator.validate(
            prompts,
            num_generations_per_prompt=int(args.num_generations_per_prompt),
            aggregate=str(args.aggregate),
        )

        stats = r.get("stats", {}) or {}
        sel_metric = str(args.selection_metric)
        metric_val = stats.get(sel_metric, None)
        try:
            metric_val_f = float(metric_val) if metric_val is not None else None
        except Exception:
            metric_val_f = None

        row = {
            "checkpoint": str(p),
            "stats": stats,
            "metadata": {
                "validated_at": datetime.now().isoformat(),
                "baseline_mlx_path": args.baseline_mlx_path if bool(args.use_mlx) else None,
                "fine_tuned_mlx_path": validator.fine_tuned_mlx_path if bool(args.use_mlx) else None,
                "generation": {
                    "temperature": float(args.generation_temperature),
                    "top_p": float(args.generation_top_p),
                    "top_k": int(args.generation_top_k),
                    "seed": (None if int(args.generation_seed) < 0 else int(args.generation_seed)),
                    "num_generations_per_prompt": int(args.num_generations_per_prompt),
                    "aggregate": str(args.aggregate),
                },
                "score_mode": str(args.score_mode),
                "selection_metric": sel_metric,
                "num_test_cases": len(prompts),
            },
        }
        results.append(row)

        if metric_val_f is not None and (best_metric_val is None or metric_val_f > best_metric_val):
            best_metric_val = metric_val_f
            best_row = row

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "best_checkpoint": (best_row.get("checkpoint") if best_row else None),
            "selection_metric": str(args.selection_metric),
            "best_metric_value": best_metric_val,
            "num_checkpoints": len(ckpts),
            "num_test_cases": len(prompts),
        },
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    logger.info(f"Wrote summary to: {out}")
    if best_row:
        stats = best_row.get("stats", {}) or {}
        logger.info(
            f"Best checkpoint: {best_row.get('checkpoint')} "
            f"({args.selection_metric}={best_metric_val}) "
            f"[abs fine_tuned={stats.get('avg_fine_tuned_score')}, "
            f"norm fine_tuned={stats.get('avg_fine_tuned_reward_normalized')}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


