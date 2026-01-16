
import re
import timeit
import time

# --- Original implementations ---

def _rubric_difficulty_components_original(prompt: str, language: str) -> dict:
    p = (prompt or "").lower()

    eff_hits = 0
    # Original:
    if re.search(r"\b\d+\s*(ms|seconds|s)\b", p):
        eff_hits += 1
    if re.search(r"\b\d+\s*(items|elements|rows|cols|nodes)\b|\bup to\b", p):
        eff_hits += 1

    correctness_hits = 0
    if re.search(r"\bconstraints?\b|\bguarantee\b|\bmust\b|\bshould\b", p):
        correctness_hits += 1

    return {"eff": eff_hits, "corr": correctness_hits}

def _strip_code_fences_original(text: str) -> str:
    import re
    t = (text or "").strip()
    t = re.sub(r"^\s*```[^\n]*\n", "", t)
    t = re.sub(r"\n```\s*$", "", t)
    return t.strip()

def _extract_score_original(text: str):
    import re
    if not text:
        return None
    cleaned = _strip_code_fences_original(text).strip()

    m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*%(?!\d)", cleaned)
    if m: return float(m.group(1)) / 100.0

    m = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?!\d)", cleaned)
    if m: return float(m.group(0))
    return None

# --- Optimized implementations ---

# Pre-compiled regexes
RE_CONSTRAINTS = re.compile(r"\bconstraints?\b|\bguarantee\b|\bmust\b|\bshould\b")
RE_TIME_UNITS = re.compile(r"\b\d+\s*(ms|seconds|s)\b")
RE_SIZE_UNITS = re.compile(r"\b\d+\s*(items|elements|rows|cols|nodes)\b|\bup to\b")
RE_FENCE_START = re.compile(r"^\s*```[^\n]*\n")
RE_FENCE_END = re.compile(r"\n```\s*$")
RE_PERCENT_SCORE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*%(?!\d)")
RE_FLOAT_SCORE = re.compile(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?!\d)")

def _rubric_difficulty_components_optimized(prompt: str, language: str) -> dict:
    p = (prompt or "").lower()

    eff_hits = 0
    if RE_TIME_UNITS.search(p):
        eff_hits += 1
    if RE_SIZE_UNITS.search(p):
        eff_hits += 1

    correctness_hits = 0
    if RE_CONSTRAINTS.search(p):
        correctness_hits += 1

    return {"eff": eff_hits, "corr": correctness_hits}

def _strip_code_fences_optimized(text: str) -> str:
    t = (text or "").strip()
    t = RE_FENCE_START.sub("", t)
    t = RE_FENCE_END.sub("", t)
    return t.strip()

def _extract_score_optimized(text: str):
    if not text:
        return None
    cleaned = _strip_code_fences_optimized(text).strip()

    m = RE_PERCENT_SCORE.search(cleaned)
    if m: return float(m.group(1)) / 100.0

    m = RE_FLOAT_SCORE.search(cleaned)
    if m: return float(m.group(0))
    return None

# --- Benchmark ---

def run_benchmark():
    prompt = "Write a function that processes up to 1000 items in 50ms. It must handle edge cases."
    text_score = "The score is 0.85"
    text_code = "```python\ndef foo(): pass\n```"

    iterations = 100000

    # Benchmark rubric
    t_rubric_orig = timeit.timeit(lambda: _rubric_difficulty_components_original(prompt, "python"), number=iterations)
    t_rubric_opt = timeit.timeit(lambda: _rubric_difficulty_components_optimized(prompt, "python"), number=iterations)

    # Benchmark extract score
    t_score_orig = timeit.timeit(lambda: _extract_score_original(text_score), number=iterations)
    t_score_opt = timeit.timeit(lambda: _extract_score_optimized(text_score), number=iterations)

    print(f"Rubric Original: {t_rubric_orig:.4f}s")
    print(f"Rubric Optimized: {t_rubric_opt:.4f}s")
    print(f"Improvement: {(t_rubric_orig - t_rubric_opt)/t_rubric_orig * 100:.2f}%")

    print(f"Score Original: {t_score_orig:.4f}s")
    print(f"Score Optimized: {t_score_opt:.4f}s")
    print(f"Improvement: {(t_score_orig - t_score_opt)/t_score_orig * 100:.2f}%")

if __name__ == "__main__":
    run_benchmark()
