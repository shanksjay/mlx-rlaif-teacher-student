
import timeit
import re
import sys
from typing import List, Dict, Optional

# --- Original Implementation ---

def _rubric_difficulty_components_original(prompt: str, language: str) -> dict[str, float]:
    """Estimate how *demanding* the prompt is along the teacher rubric dimensions.

    Returns values in [0,1] (higher = more demanding).

    This is intentionally a lightweight heuristic (keyword/constraint based) so it can run in the hot path.
    """
    p = (prompt or "").lower()
    lg = (language or "python").lower()

    def has_any(words: list[str]) -> bool:
        return any(w in p for w in words)

    def count_any(words: list[str]) -> int:
        return sum(1 for w in words if w in p)

    # Correctness: multi-part requirements, edge cases, concurrency/safety, parsing, etc.
    correctness_hits = 0
    correctness_hits += count_any(
        [
            "edge case",
            "corner case",
            "validate",
            "invalid",
            "error handling",
            "robust",
            "safely",
            "thread-safe",
            "thread safe",
            "lock-free",
            "deadlock",
            "race",
            "atomic",
            "parse",
            "parser",
            "serialize",
            "deserialize",
            "json",
            "unicode",
            "overflow",
            "underflow",
            "null",
            "nullptr",
        ]
    )
    # Presence of constraints section-like patterns increases correctness demand.
    if re.search(r"\bconstraints?\b|\bguarantee\b|\bmust\b|\bshould\b", p):
        correctness_hits += 1
    correctness = min(1.0, correctness_hits / 6.0)

    # Code quality: API design, patterns, RAII, clean architecture, tests.
    quality_hits = 0
    quality_hits += count_any(
        [
            "clean",
            "readable",
            "well-structured",
            "well structured",
            "maintainable",
            "refactor",
            "design pattern",
            "singleton",
            "raii",
            "interface",
            "abstraction",
            "encapsulation",
            "modular",
            "unit test",
            "tests",
        ]
    )
    # If prompt asks for "class" or "library" style implementation, quality demands rise.
    if has_any(["class ", "api", "library", "module"]):
        quality_hits += 1
    quality = min(1.0, quality_hits / 6.0)

    # Efficiency: performance constraints, complexity, optimization, large input.
    eff_hits = 0
    eff_hits += count_any(
        [
            "efficient",
            "optimize",
            "performance",
            "fast",
            "low latency",
            "high throughput",
            "big-o",
            "o(",
            "time complexity",
            "space complexity",
            "memory",
            "constant time",
            "log n",
            "n log n",
            "linear time",
        ]
    )
    if re.search(r"\b\d+\s*(ms|seconds|s)\b", p):
        eff_hits += 1
    if re.search(r"\b\d+\s*(items|elements|rows|cols|nodes)\b|\bup to\b", p):
        eff_hits += 1
    efficiency = min(1.0, eff_hits / 6.0)

    # Documentation: explicit documentation/comments, examples.
    doc_hits = 0
    doc_hits += count_any(
        [
            "document",
            "documentation",
            "docstring",
            "comments",
            "commented",
            "well-documented",
            "well documented",
            "explain",
            "explanation",
            "examples",
        ]
    )
    documentation = min(1.0, doc_hits / 4.0)

    # Composite rubric demand (mirrors scoring weights)
    demand = (0.3 * correctness) + (0.3 * quality) + (0.2 * efficiency) + (0.2 * documentation)

    # Language base multiplier: cpp/rust typically have more incidental complexity.
    if lg in ("cpp", "c++"):
        lang_weight = 1.10
    elif lg == "rust":
        lang_weight = 1.15
    else:
        lang_weight = 1.00

    return {
        "correctness": float(correctness),
        "code_quality": float(quality),
        "efficiency": float(efficiency),
        "documentation": float(documentation),
        "rubric_demand": float(min(1.0, max(0.0, demand))),
        "lang_weight": float(lang_weight),
    }

# --- Optimized Implementation ---

# Compile regexes at module level
_CORRECTNESS_KEYWORDS = [
    "edge case", "corner case", "validate", "invalid", "error handling",
    "robust", "safely", "thread-safe", "thread safe", "lock-free",
    "deadlock", "race", "atomic", "parse", "parser", "serialize",
    "deserialize", "json", "unicode", "overflow", "underflow",
    "null", "nullptr"
]
# Create regex pattern for correctness keywords
# Use escape to handle potential special chars, join with |
_CORRECTNESS_PATTERN = re.compile("|".join(map(re.escape, _CORRECTNESS_KEYWORDS)))

_QUALITY_KEYWORDS = [
    "clean", "readable", "well-structured", "well structured",
    "maintainable", "refactor", "design pattern", "singleton",
    "raii", "interface", "abstraction", "encapsulation",
    "modular", "unit test", "tests"
]
_QUALITY_PATTERN = re.compile("|".join(map(re.escape, _QUALITY_KEYWORDS)))

_QUALITY_EXTRA_KEYWORDS = ["class ", "api", "library", "module"]
_QUALITY_EXTRA_PATTERN = re.compile("|".join(map(re.escape, _QUALITY_EXTRA_KEYWORDS)))

_EFFICIENCY_KEYWORDS = [
    "efficient", "optimize", "performance", "fast", "low latency",
    "high throughput", "big-o", "o(", "time complexity",
    "space complexity", "memory", "constant time", "log n",
    "n log n", "linear time"
]
_EFFICIENCY_PATTERN = re.compile("|".join(map(re.escape, _EFFICIENCY_KEYWORDS)))

_DOCUMENTATION_KEYWORDS = [
    "document", "documentation", "docstring", "comments",
    "commented", "well-documented", "well documented",
    "explain", "explanation", "examples"
]
_DOCUMENTATION_PATTERN = re.compile("|".join(map(re.escape, _DOCUMENTATION_KEYWORDS)))

_CONSTRAINTS_PATTERN = re.compile(r"\bconstraints?\b|\bguarantee\b|\bmust\b|\bshould\b")
_EFFICIENCY_TIME_PATTERN = re.compile(r"\b\d+\s*(ms|seconds|s)\b")
_EFFICIENCY_SIZE_PATTERN = re.compile(r"\b\d+\s*(items|elements|rows|cols|nodes)\b|\bup to\b")

def _rubric_difficulty_components_optimized(prompt: str, language: str) -> dict[str, float]:
    p = (prompt or "").lower()
    lg = (language or "python").lower()

    # Using len(findall) counts non-overlapping occurrences.
    # The original implementation counted presence of *each keyword*.
    # Original: sum(1 for w in words if w in p)
    # This means if "thread-safe" and "thread safe" both appear, and prompt has "thread-safe",
    # original counts 1 (if "thread safe" matches substring "thread-safe" depends on implementation, but `in` does).
    # Wait, `in` matches substring.
    # "thread safe" in "thread-safe"? No.
    # "race" in "trace"? Yes.

    # Original logic: sum(1 for w in words if w in p)
    # This loops over ALL words and checks if they are in p.
    # If p="trace", "race" is in p -> count=1.

    # If we use regex findall, we find occurrences.
    # But we want to count how many *unique keywords* matched?
    # Or just total matches?
    # Original: `sum(1 ...)` counts how many keywords from the list are present.
    # So if "race" appears twice, it's counted once (because we iterate list of keywords).
    # If "race" and "atomic" both appear, it counts 2.

    # Optimized approach 1: Loop over cached list (avoids list creation)
    # This is safer to preserve exact behavior.

    correctness_hits = sum(1 for w in _CORRECTNESS_KEYWORDS if w in p)

    if _CONSTRAINTS_PATTERN.search(p):
        correctness_hits += 1
    correctness = min(1.0, correctness_hits / 6.0)

    quality_hits = sum(1 for w in _QUALITY_KEYWORDS if w in p)
    if any(w in p for w in _QUALITY_EXTRA_KEYWORDS):
         quality_hits += 1
    quality = min(1.0, quality_hits / 6.0)

    eff_hits = sum(1 for w in _EFFICIENCY_KEYWORDS if w in p)

    if _EFFICIENCY_TIME_PATTERN.search(p):
        eff_hits += 1
    if _EFFICIENCY_SIZE_PATTERN.search(p):
        eff_hits += 1
    efficiency = min(1.0, eff_hits / 6.0)

    doc_hits = sum(1 for w in _DOCUMENTATION_KEYWORDS if w in p)
    documentation = min(1.0, doc_hits / 4.0)

    demand = (0.3 * correctness) + (0.3 * quality) + (0.2 * efficiency) + (0.2 * documentation)

    if lg in ("cpp", "c++"):
        lang_weight = 1.10
    elif lg == "rust":
        lang_weight = 1.15
    else:
        lang_weight = 1.00

    return {
        "correctness": float(correctness),
        "code_quality": float(quality),
        "efficiency": float(efficiency),
        "documentation": float(documentation),
        "rubric_demand": float(min(1.0, max(0.0, demand))),
        "lang_weight": float(lang_weight),
    }

# --- Optimized Implementation v2 (Regex Counting) ---
# Actually `w in p` is fast in CPython. But iterating a list 4 times per call might be slower than
# just regex matching if optimized correctly, OR regex might be slower for many keywords.
# Let's test checking cached lists vs re-creating lists vs regex.

# Optimization v2: just pre-define lists (what I did in v1 above).
# I suspect the list creation in original code is the main overhead.
# Let's verify.

# --- Benchmark ---

prompts = [
    "Write a thread-safe python function to parse json safely and handle edge cases.",
    "Optimize this O(n^2) algorithm to O(n log n) for high throughput.",
    "Create a clean, readable class with unit tests and documentation.",
    "Simple function to add two numbers.",
    "Implement a lock-free queue in Rust with atomic operations to avoid deadlocks and race conditions." * 10, # Longer prompt
] * 20

def run_original():
    for p in prompts:
        _rubric_difficulty_components_original(p, "python")

def run_optimized():
    for p in prompts:
        _rubric_difficulty_components_optimized(p, "python")

if __name__ == "__main__":
    print(f"Benchmarking with {len(prompts)} prompts...")

    t_orig = timeit.timeit(run_original, number=100)
    print(f"Original: {t_orig:.4f}s")

    t_opt = timeit.timeit(run_optimized, number=100)
    print(f"Optimized: {t_opt:.4f}s")

    print(f"Speedup: {t_orig / t_opt:.2f}x")

    # Verify correctness
    p1 = prompts[0]
    res_orig = _rubric_difficulty_components_original(p1, "python")
    res_opt = _rubric_difficulty_components_optimized(p1, "python")

    assert res_orig == res_opt, f"Results differ!\nOriginal: {res_orig}\nOptimized: {res_opt}"
    print("Verification passed.")


# --- Benchmarking Score Extraction ---

def _strip_code_fences_original(text: str) -> str:
    import re
    t = (text or "").strip()
    t = re.sub(r"^\s*```[^\n]*\n", "", t)
    t = re.sub(r"\n```\s*$", "", t)
    return t.strip()

def _extract_score_original(text: str) -> Optional[float]:
    import re
    if not text:
        return None
    cleaned = _strip_code_fences_original(text).strip()
    first_tok = cleaned.split()[0].strip() if cleaned.split() else cleaned
    for candidate in (first_tok, cleaned):
        try:
            v = float(candidate)
            if 0.0 <= v <= 1.0:
                return v
        except Exception:
            pass
    m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*%(?!\d)", cleaned)
    if m:
        try:
            v = float(m.group(1)) / 100.0
            if 0.0 <= v <= 1.0:
                return v
        except Exception:
            pass
    m = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?!\d)", cleaned)
    if m:
        try:
            v = float(m.group(0))
            if 0.0 <= v <= 1.0:
                return v
        except Exception:
            pass
    return None

# Optimized versions
_FENCE_START = re.compile(r"^\s*```[^\n]*\n")
_FENCE_END = re.compile(r"\n```\s*$")
_PERCENT_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*%(?!\d)")
_FLOAT_PATTERN = re.compile(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?!\d)")

def _strip_code_fences_optimized(text: str) -> str:
    t = (text or "").strip()
    t = _FENCE_START.sub("", t)
    t = _FENCE_END.sub("", t)
    return t.strip()

def _extract_score_optimized(text: str) -> Optional[float]:
    if not text:
        return None
    cleaned = _strip_code_fences_optimized(text).strip()
    first_tok = cleaned.split()[0].strip() if cleaned.split() else cleaned
    for candidate in (first_tok, cleaned):
        try:
            v = float(candidate)
            if 0.0 <= v <= 1.0:
                return v
        except Exception:
            pass
    m = _PERCENT_PATTERN.search(cleaned)
    if m:
        try:
            v = float(m.group(1)) / 100.0
            if 0.0 <= v <= 1.0:
                return v
        except Exception:
            pass
    m = _FLOAT_PATTERN.search(cleaned)
    if m:
        try:
            v = float(m.group(0))
            if 0.0 <= v <= 1.0:
                return v
        except Exception:
            pass
    return None

score_texts = [
    "0.75",
    "Score: 0.8",
    "The code is good. 95%",
    "```python\n0.5\n```",
    "0.123",
    "Invalid score",
    "1.0",
    "0.0",
] * 100

def run_score_original():
    for t in score_texts:
        _extract_score_original(t)

def run_score_optimized():
    for t in score_texts:
        _extract_score_optimized(t)

if __name__ == "__main__":
    print(f"\nBenchmarking Score Extraction with {len(score_texts)} texts...")
    t_score_orig = timeit.timeit(run_score_original, number=100)
    print(f"Original: {t_score_orig:.4f}s")
    t_score_opt = timeit.timeit(run_score_optimized, number=100)
    print(f"Optimized: {t_score_opt:.4f}s")
    print(f"Speedup: {t_score_orig / t_score_opt:.2f}x")

    # Verify
    for t in score_texts[:10]:
        assert _extract_score_original(t) == _extract_score_optimized(t)
    print("Verification passed.")
