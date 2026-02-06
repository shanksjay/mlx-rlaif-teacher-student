## 2024-05-22 - Redundant Tokenization in Dataset
**Learning:** `CodeDataset` in `train_rlaif.py` was tokenizing the prompt in `__getitem__`, but the training loop (`train` and `generate_student_samples`) only used the raw prompt string. The tokenized `input_ids` and `attention_mask` were discarded or unused, wasting significant CPU time (up to 1000x slower dataset iteration in benchmarks).
**Action:** Removed tokenization from `CodeDataset.__getitem__`. Always verify if data loader outputs are actually used by the consumer before performing expensive preprocessing.

## 2024-05-22 - Safe Caching of Mutable Objects
**Learning:** `_rubric_difficulty_components` in `train_rlaif.py` returns a mutable dictionary. Simply caching it with `@lru_cache` is dangerous because callers might modify the returned dictionary, polluting the cache for future calls.
**Action:** When caching functions that return mutable objects (dicts, lists), use a two-layer approach: 1) A cached implementation function (`_impl`) that returns the object, and 2) A wrapper function that calls `_impl(...).copy()` to return a safe, independent copy to the caller. This incurs a small copy cost but guarantees correctness.

## 2024-05-22 - Non-blocking System Metrics
**Learning:** `psutil.cpu_percent(interval=0.1)` blocks the calling thread for the specified interval. When used in a training loop (e.g., for logging), this adds unnecessary latency (e.g., 0.2s if called twice).
**Action:** Use `psutil.cpu_percent(interval=None)` for non-blocking calls. This returns the CPU usage since the last call. Be aware that the first call returns 0.0, which is acceptable for periodic logging.
