## 2024-05-22 - Safe Caching of Mutable Objects
**Learning:** `_rubric_difficulty_components` in `train_rlaif.py` returns a mutable dictionary. Simply caching it with `@lru_cache` is dangerous because callers might modify the returned dictionary, polluting the cache for future calls.
**Action:** When caching functions that return mutable objects (dicts, lists), use a two-layer approach: 1) A cached implementation function (`_impl`) that returns the object, and 2) A wrapper function that calls `_impl(...).copy()` to return a safe, independent copy to the caller. This incurs a small copy cost but guarantees correctness.

## 2024-05-22 - Non-blocking System Metrics
**Learning:** `psutil.cpu_percent(interval=0.1)` blocks the calling thread for the specified interval. When used in a training loop (e.g., for logging), this adds unnecessary latency (e.g., 0.2s if called twice).
**Action:** Use `psutil.cpu_percent(interval=None)` for non-blocking calls. This returns the CPU usage since the last call. Be aware that the first call returns 0.0, which is acceptable for periodic logging.

## 2024-05-22 - Avoid Premature Tokenization in Datasets
**Learning:** `CodeDataset` in `train_rlaif.py` was tokenizing prompts in `__getitem__`, but the training loop only used the raw prompt strings for generation (RLAIF) and later re-tokenized the full (prompt + response) sequence. This initial tokenization was 100% wasted CPU work (~170x slowdown in data loading benchmarks).
**Action:** When working with generation-heavy pipelines (like RLAIF/RLHF), check if the `Dataset` outputs are actually used by the model directly. If the loop handles generation/sampling first, defer tokenization until the final training batch is constructed to avoid double work.
