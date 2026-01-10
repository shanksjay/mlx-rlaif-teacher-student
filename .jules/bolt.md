## 2024-05-23 - [Optimization: O(1) Cache Expiration Check]
**Learning:** LRU cache expiration checks can become a performance bottleneck if implemented as O(N) iteration over the entire cache, especially for large caches. Using `itertools.islice` to check only a bounded number of the oldest (LRU) items transforms this into an O(1) (or O(k)) operation.
**Action:** When implementing expiration logic for ordered caches, always use bounded iteration from the LRU end rather than full scans.
