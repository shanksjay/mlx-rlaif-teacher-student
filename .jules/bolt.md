## 2024-03-21 - [Optimized cache cleanup]
**Learning:** Cleaning up large caches by iterating over a full `list(.items())` copy is a major performance bottleneck in Python.
**Action:** Use `itertools.islice()` to iterate over a limited number of items (the oldest ones in an `OrderedDict`/`dict` LRU) to amortize cleanup costs to O(k) instead of O(N).
