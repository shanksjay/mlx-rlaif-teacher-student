## 2024-05-22 - Safe Caching of Mutable Objects
**Learning:** `_rubric_difficulty_components` in `train_rlaif.py` returns a mutable dictionary. Simply caching it with `@lru_cache` is dangerous because callers might modify the returned dictionary, polluting the cache for future calls.
**Action:** When caching functions that return mutable objects (dicts, lists), use a two-layer approach: 1) A cached implementation function (`_impl`) that returns the object, and 2) A wrapper function that calls `_impl(...).copy()` to return a safe, independent copy to the caller. This incurs a small copy cost but guarantees correctness.

## 2024-05-22 - Non-blocking System Metrics
**Learning:** `psutil.cpu_percent(interval=0.1)` blocks the calling thread for the specified interval. When used in a training loop (e.g., for logging), this adds unnecessary latency (e.g., 0.2s if called twice).
**Action:** Use `psutil.cpu_percent(interval=None)` for non-blocking calls. This returns the CPU usage since the last call. Be aware that the first call returns 0.0, which is acceptable for periodic logging.

## 2026-02-03 - Expensive Monitoring in Hot Loop
**Learning:** `train_rlaif.py` was calling `_capture_parameter_state()` (cloning all model parameters) and `_compute_parameter_changes()` on *every* optimizer step to track updates. This caused massive overhead due to memory allocation, copy, and CPU-GPU synchronization (`.item()` calls), reducing training throughput significantly (simulated ~4x slowdown).
**Action:** Gate expensive monitoring/debugging checks behind `if step % logging_steps == 0`. Ensure monitoring code (like parameter diffing or gradient norm checks) runs only when necessary for logging, not on every iteration.
