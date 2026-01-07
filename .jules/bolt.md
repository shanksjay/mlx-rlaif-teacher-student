## 2024-05-24 - [LRU Cache Iteration Optimization]
**Learning:** In LRU cache implementations using `OrderedDict`, iterating over the entire dictionary to find expired items can be O(N), which is expensive if the cache is large.
**Action:** Since `OrderedDict` maintains order, we can iterate from the beginning (LRU end) and limit the checks to a small batch (e.g., 100 items). This makes cleanup effectively O(1) while still catching expired items that have drifted to the LRU end.
