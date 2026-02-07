## 2024-05-22 - Safe Caching of Mutable Objects
**Learning:** `_rubric_difficulty_components` in `train_rlaif.py` returns a mutable dictionary. Simply caching it with `@lru_cache` is dangerous because callers might modify the returned dictionary, polluting the cache for future calls.
**Action:** When caching functions that return mutable objects (dicts, lists), use a two-layer approach: 1) A cached implementation function (`_impl`) that returns the object, and 2) A wrapper function that calls `_impl(...).copy()` to return a safe, independent copy to the caller. This incurs a small copy cost but guarantees correctness.

## 2024-05-22 - Non-blocking System Metrics
**Learning:** `psutil.cpu_percent(interval=0.1)` blocks the calling thread for the specified interval. When used in a training loop (e.g., for logging), this adds unnecessary latency (e.g., 0.2s if called twice).
**Action:** Use `psutil.cpu_percent(interval=None)` for non-blocking calls. This returns the CPU usage since the last call. Be aware that the first call returns 0.0, which is acceptable for periodic logging.

## 2026-02-07 - Redundant Dataset Tokenization
**Learning:** `CodeDataset` in `train_rlaif.py` was performing expensive tokenization (returning `input_ids`, `attention_mask`) on every access, but the training loop only used the raw text (`prompt`, `language`) for generation. The training step later re-tokenized the full sequence (prompt + completion), making the initial tokenization entirely redundant.
**Action:** For generation-heavy workflows (RLHF/RLAIF), ensure the dataset returns only the raw text needed for generation, avoiding premature and unused tokenization.
