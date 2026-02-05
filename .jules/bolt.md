## 2024-05-22 - Safe Caching of Mutable Objects
**Learning:** `_rubric_difficulty_components` in `train_rlaif.py` returns a mutable dictionary. Simply caching it with `@lru_cache` is dangerous because callers might modify the returned dictionary, polluting the cache for future calls.
**Action:** When caching functions that return mutable objects (dicts, lists), use a two-layer approach: 1) A cached implementation function (`_impl`) that returns the object, and 2) A wrapper function that calls `_impl(...).copy()` to return a safe, independent copy to the caller. This incurs a small copy cost but guarantees correctness.

## 2024-05-22 - Non-blocking System Metrics
**Learning:** `psutil.cpu_percent(interval=0.1)` blocks the calling thread for the specified interval. When used in a training loop (e.g., for logging), this adds unnecessary latency (e.g., 0.2s if called twice).
**Action:** Use `psutil.cpu_percent(interval=None)` for non-blocking calls. This returns the CPU usage since the last call. Be aware that the first call returns 0.0, which is acceptable for periodic logging.

## 2024-05-23 - Avoiding Redundant Tokenization in Datasets
**Learning:** In hybrid RLAIF loops (like `train_rlaif.py`), the dataset often tokenizes prompts during loading (`__getitem__`), but the training loop extracts raw prompts for generation (which re-tokenizes them) and then tokenizes the final (prompt+code) sequence for training. The initial tokenization in `Dataset` is completely wasted CPU/memory.
**Action:** Verify if `input_ids` from the dataset are actually used in the training loop. If not (and only raw text is used), remove the tokenization from `__getitem__`. This can yield massive speedups (>1000x for data iteration) by avoiding tensor creation overhead. Be careful to check for side consumers (like logging or correlation analysis) and ensure they have fallbacks (e.g. char length vs token length).
