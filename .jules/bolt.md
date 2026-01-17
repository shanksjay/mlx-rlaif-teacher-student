## 2024-05-22 - Regex and Keyword Optimization in Scoring Hot Path
**Learning:** `_rubric_difficulty_components` in `train_rlaif.py` is a hot path for scoring. It was reconstructing keyword lists and regex patterns on every call. Moving these to module-level constants and pre-compiling regexes yielded a ~9% speedup for the function.
**Action:** For hot-path functions in python scripts (especially those called per-sample), ensure constants and regexes are defined at module level to avoid reconstruction overhead.
