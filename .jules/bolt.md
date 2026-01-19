## 2024-01-19 - Regex Compilation Performance
**Learning:** Pre-compiling regex patterns (`re.compile`) at the module level in hot-path functions like `_rubric_difficulty_components` and `_extract_score` provided significant performance gains (14% and 38% respectively) compared to compiling inside the function.
**Action:** When implementing heuristic functions or text processing utilities that use regex, always extract static patterns to module-level constants. This is especially critical in training loops where these functions are called thousands of times.
