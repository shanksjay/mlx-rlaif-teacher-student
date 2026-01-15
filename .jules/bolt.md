## 2024-03-24 - Efficient Dictionary Iteration in Python
**Learning:** Iterating over `dict.items()` directly (a view) is significantly faster (30-40%) and more memory efficient than `list(dict.items())` for large dictionaries, provided the dictionary structure is not modified *during* the iteration.
**Action:** When collecting keys for removal or processing dictionary items, iterate directly over `.items()` or `keys()` and defer modification to a second pass, rather than snapshotting the entire dictionary with `list()`.
