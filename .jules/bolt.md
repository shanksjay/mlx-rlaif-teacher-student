## 2024-03-24 - Efficient Dictionary Iteration in Python
**Learning:** Iterating over `dict.items()` directly (a view) is significantly faster (30-40%) and more memory efficient than `list(dict.items())` for large dictionaries, provided the dictionary structure is not modified *during* the iteration.
**Action:** When collecting keys for removal or processing dictionary items, iterate directly over `.items()` or `keys()` and defer modification to a second pass, rather than snapshotting the entire dictionary with `list()`.
## 2024-01-19 - Regex Compilation Performance
**Learning:** Pre-compiling regex patterns (`re.compile`) at the module level in hot-path functions like `_rubric_difficulty_components` and `_extract_score` provided significant performance gains (14% and 38% respectively) compared to compiling inside the function.
**Action:** When implementing heuristic functions or text processing utilities that use regex, always extract static patterns to module-level constants. This is especially critical in training loops where these functions are called thousands of times.
