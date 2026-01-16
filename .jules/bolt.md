## 2024-05-22 - Regex Compilation in Hot Paths
**Learning:** `scripts/training/train_rlaif.py` contains monolithic training logic where helper functions like `_extract_score` are called per-sample. Inline regex compilation (`re.search`) in these hot paths adds significant overhead (~25-44%).
**Action:** Extract regex patterns to module-level constants and use `re.compile`. Refactor nested helper functions to module level to enable proper unit testing and optimization verification.
