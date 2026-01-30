## 2024-05-22 - Safe Caching of Mutable Objects
**Learning:** `_rubric_difficulty_components` in `train_rlaif.py` returns a mutable dictionary. Simply caching it with `@lru_cache` is dangerous because callers might modify the returned dictionary, polluting the cache for future calls.
**Action:** When caching functions that return mutable objects (dicts, lists), use a two-layer approach: 1) A cached implementation function (`_impl`) that returns the object, and 2) A wrapper function that calls `_impl(...).copy()` to return a safe, independent copy to the caller. This incurs a small copy cost but guarantees correctness.

## 2024-05-22 - Non-blocking System Metrics
**Learning:** `psutil.cpu_percent(interval=0.1)` blocks the calling thread for the specified interval. When used in a training loop (e.g., for logging), this adds unnecessary latency (e.g., 0.2s if called twice).
**Action:** Use `psutil.cpu_percent(interval=None)` for non-blocking calls. This returns the CPU usage since the last call. Be aware that the first call returns 0.0, which is acceptable for periodic logging.

## 2024-05-22 - Avoid Double Tokenization in RLAIF Datasets
**Learning:** In RLAIF pipelines, the dataset often serves merely as a prompt provider for generation. Tokenizing in `__getitem__` is wasteful if the generation step (e.g., MLX or PyTorch generate) re-tokenizes the prompt anyway. In this codebase, `CodeDataset` tokenization was redundant and blocking the main thread (num_workers=0).
**Action:** Check if `input_ids` from Dataset are actually used. If the pipeline generates from raw text, return raw text from Dataset and skip tokenization. This yielded a >1000x speedup in dataset iteration.
