## 2024-05-21 - OrderedDict Cache Optimization
**Learning:** `OrderedDict` in `train_rlaif.py` is used as an LRU cache. Iterating over `list(cache.items())` creates an unnecessary O(N) copy. Iterating over `cache.items()` directly is safe if the loop does not modify the dictionary structure (e.g. collecting keys to delete later).
**Action:** Avoid `list()` copy when iterating over dictionary items for inspection/filtering.
