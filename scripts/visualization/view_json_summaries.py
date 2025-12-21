#!/usr/bin/env python3
"""
Offline reader for JSONL training summaries produced by `train_rlaif.py`.

Reads:
- <dir>/batches.jsonl
- <dir>/epochs.jsonl

Examples:
  uv run python scripts/visualization/view_json_summaries.py --dir ./logs/json_summaries --type batch --tail 20
  uv run python scripts/visualization/view_json_summaries.py --dir ./logs/json_summaries --type epoch
  uv run python scripts/visualization/view_json_summaries.py --dir ./logs/json_summaries --type batch --csv-out /tmp/batches.csv
"""

import argparse
import json
import os
from typing import Any, Dict, List


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        kk = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, kk))
        else:
            out[kk] = v
    return out


def _print_rows(rows: List[Dict[str, Any]], cols: List[str], tail: int | None = None):
    if tail is not None and tail > 0:
        rows = rows[-tail:]
    # Build table
    lines = []
    header = " | ".join(cols)
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        fr = _flatten(r)
        vals = []
        for c in cols:
            v = fr.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.4g}")
            else:
                vals.append(str(v))
        lines.append(" | ".join(vals))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="View offline JSONL batch/epoch summaries as time series.")
    ap.add_argument("--dir", type=str, default="./logs/json_summaries", help="Directory containing batches.jsonl/epochs.jsonl")
    ap.add_argument("--type", type=str, choices=["batch", "epoch"], default="batch", help="Which time series to view")
    ap.add_argument("--tail", type=int, default=50, help="Show last N rows (0 = show all)")
    ap.add_argument("--csv-out", type=str, default=None, help="Optional CSV output path")
    args = ap.parse_args()

    jsonl_path = os.path.join(args.dir, "batches.jsonl" if args.type == "batch" else "epochs.jsonl")
    rows = _read_jsonl(jsonl_path)
    if not rows:
        raise SystemExit(f"No rows found in {jsonl_path}. Did you enable logging.save_json_summaries?")

    if args.type == "batch":
        cols = [
            "ts_iso",
            "epoch",
            "batch_idx",
            "global_step",
            "batch_size",
            "generation_backend",
            "timing_s.generation",
            "timing_s.scoring",
            "timing_s.training",
            "timing_s.total",
            "tokens.gen_raw",
            "tokens.gen_kept",
            "throughput_tok_s.gen_raw_overall",
            "throughput_tok_s.gen_kept_overall",
            "throughput_tok_s.training",
            "samples.dup_filtered",
            "samples.diversity_ratio",
            "rewards.mean",
            "reward_gain.gain_from_baseline",
            "reward_gain.gain_vs_prev_batch",
            "reward_gain.ema_reward",
            "reward_gain.ema_gain_from_baseline",
            "loss.loss",
        ]
    else:
        cols = [
            "ts_iso",
            "epoch",
            "generation_backend",
            "epoch_time_s",
            "avg_reward",
            "avg_loss",
            "reward_trend",
            "loss_trend",
            "api.calls",
            "api.cache_hit_rate",
            "diversity.unique_ratio",
            "performance.generation.tok_s_raw",
            "performance.generation.tok_s_kept",
            "performance.training.tok_s",
        ]

    tail = None if args.tail == 0 else args.tail
    _print_rows(rows, cols, tail=tail)

    if args.csv_out:
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise SystemExit(f"pandas is required for CSV export: {e}")
        df = pd.DataFrame([_flatten(r) for r in rows])
        df.to_csv(args.csv_out, index=False)
        print(f"\nWrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()


