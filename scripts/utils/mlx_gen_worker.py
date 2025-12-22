#!/usr/bin/env python3
"""
MLX generation worker (subprocess) to isolate MLX Metal allocations from PyTorch MPS.

Protocol: JSON Lines over stdin/stdout.

Request:
  {"id": 1, "op": "generate", "prompt": "...", "max_tokens": 128,
   "do_sample": false, "temp": 1.0, "top_p": 0.95, "top_k": 50}

Response:
  {"id": 1, "ok": true, "text": "...", "seconds": 0.42, "output_tokens": 123}

Supports:
  - op=ping
  - op=shutdown
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, Optional


def _jsonl_write(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="MLX generation worker (JSONL stdin/stdout)")
    parser.add_argument("--model-path", required=True, help="Path to MLX model directory (mlx_lm.load compatible)")
    args = parser.parse_args()

    try:
        import mlx.core as mx  # noqa: F401
        from mlx_lm import load, generate as mlx_generate
    except Exception as e:
        _jsonl_write({"id": -1, "ok": False, "error": f"MLX import failed: {type(e).__name__}: {e}"})
        return 2

    try:
        model, tokenizer = load(args.model_path)
    except Exception as e:
        _jsonl_write({"id": -1, "ok": False, "error": f"mlx_lm.load failed: {type(e).__name__}: {e}"})
        return 3

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            _jsonl_write({"id": -1, "ok": False, "error": "invalid_json"})
            continue

        req_id = req.get("id", -1)
        op = req.get("op")
        if op == "ping":
            _jsonl_write({"id": req_id, "ok": True, "pong": True})
            continue
        if op == "shutdown":
            _jsonl_write({"id": req_id, "ok": True, "shutdown": True})
            return 0
        if op != "generate":
            _jsonl_write({"id": req_id, "ok": False, "error": f"unknown_op:{op}"})
            continue

        prompt = req.get("prompt", "")
        prompt_ids = req.get("prompt_ids")
        max_tokens = int(req.get("max_tokens", 128))
        do_sample = bool(req.get("do_sample", False))
        temp = float(req.get("temp", 1.0))
        top_p = float(req.get("top_p", 0.95))
        top_k = int(req.get("top_k", 50))

        sampler = None
        if do_sample:
            try:
                # mlx-lm >= 0.29 uses sample_utils.make_sampler
                from mlx_lm.sample_utils import make_sampler  # type: ignore
                sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
            except Exception:
                sampler = None
                do_sample = False

        # Prefer token IDs prompt to avoid tokenizer overhead in mlx-lm
        used_prompt = prompt_ids if isinstance(prompt_ids, list) and prompt_ids else prompt
        t0 = time.time()
        try:
            text = mlx_generate(
                model,
                tokenizer,
                prompt=used_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            )
            dt = time.time() - t0
            # Approximate output token count for downstream throughput metrics.
            # If prompt was provided as token IDs, we can't reliably string-strip; return 0.
            out_tokens = 0
            if isinstance(text, str) and isinstance(prompt, str) and prompt and text.startswith(prompt):
                gen = text[len(prompt):].strip()
                try:
                    out_tokens = len(tokenizer.encode(gen, add_special_tokens=False))
                except Exception:
                    out_tokens = 0
            _jsonl_write(
                {
                    "id": req_id,
                    "ok": True,
                    "text": text,
                    "seconds": dt,
                    "output_tokens": out_tokens,
                    "did_sample": bool(do_sample and sampler is not None),
                    "pid": int(__import__("os").getpid()),
                    # MLX Metal memory stats (best-effort)
                    "mlx_active_bytes": int(mx.get_active_memory()) if hasattr(mx, "get_active_memory") else 0,
                    "mlx_cache_bytes": int(mx.get_cache_memory()) if hasattr(mx, "get_cache_memory") else 0,
                    "mlx_peak_bytes": int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else 0,
                }
            )
        except Exception as e:
            dt = time.time() - t0
            _jsonl_write({"id": req_id, "ok": False, "error": f"{type(e).__name__}: {e}", "seconds": dt})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


