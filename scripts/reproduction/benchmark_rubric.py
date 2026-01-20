
import time
import re
import statistics
from typing import Optional, Dict

# --- Original Implementation (Simulated) ---

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

def _strip_code_fences_original(text: str) -> str:
    """Remove surrounding ```lang ... ``` fences if present."""
    import re
    t = (text or "").strip()
    # Remove a leading fence line like ``` or ```python
    t = re.sub(r"^\s*```[^\n]*\n", "", t)
    # Remove a trailing fence
    t = re.sub(r"\n```\s*$", "", t)
    return t.strip()

def _extract_score_original(text: str) -> Optional[float]:
    """Best-effort extraction of a float score in [0,1] from a teacher response."""
    import re
    if not text:
        return None
    cleaned = _strip_code_fences_original(text).strip()
    # First token often is the score; this avoids picking up rubric numbers if the model misbehaves.
    first_tok = cleaned.split()[0].strip() if cleaned.split() else cleaned
    for candidate in (first_tok, cleaned):
        try:
            v = float(candidate)
            if 0.0 <= v <= 1.0:
                return v
        except Exception:
            pass
    # Percent form like "75%" -> 0.75
    m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*%(?!\d)", cleaned)
    if m:
        try:
            v = float(m.group(1)) / 100.0
            if 0.0 <= v <= 1.0:
                return v
        except Exception:
            pass
    # Float in [0,1], including ".75"
    m = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?!\d)", cleaned)
    if m:
        try:
            v = float(m.group(0))
            if 0.0 <= v <= 1.0:
                return v
        except Exception:
            pass
    return None

# --- Optimized Implementation (Proposed) ---

# Module-level pre-compiled regexes
_RE_CORRECTNESS_KEYWORDS = [
    "edge case", "corner case", "validate", "invalid", "error handling", "robust",
    "safely", "thread-safe", "thread safe", "lock-free", "deadlock", "race",
    "atomic", "parse", "parser", "serialize", "deserialize", "json", "unicode",
    "overflow", "underflow", "null", "nullptr"
]
_RE_QUALITY_KEYWORDS = [
    "clean", "readable", "well-structured", "well structured", "maintainable",
    "refactor", "design pattern", "singleton", "raii", "interface", "abstraction",
    "encapsulation", "modular", "unit test", "tests"
]
_RE_EFFICIENCY_KEYWORDS = [
    "efficient", "optimize", "performance", "fast", "low latency", "high throughput",
    "big-o", "o(", "time complexity", "space complexity", "memory", "constant time",
    "log n", "n log n", "linear time"
]
_RE_DOCS_KEYWORDS = [
    "document", "documentation", "docstring", "comments", "commented",
    "well-documented", "well documented", "explain", "explanation", "examples"
]

_RE_CONSTRAINTS = re.compile(r"\bconstraints?\b|\bguarantee\b|\bmust\b|\bshould\b")
_RE_TIME_UNITS = re.compile(r"\b\d+\s*(ms|seconds|s)\b")
_RE_SIZE_UNITS = re.compile(r"\b\d+\s*(items|elements|rows|cols|nodes)\b|\bup to\b")
_RE_FENCE_START = re.compile(r"^\s*```[^\n]*\n")
_RE_FENCE_END = re.compile(r"\n```\s*$")
_RE_PERCENT = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*%(?!\d)")
_RE_FLOAT = re.compile(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?!\d)")

def _rubric_difficulty_components_optimized(prompt: str, language: str) -> dict[str, float]:
    p = (prompt or "").lower()
    lg = (language or "python").lower()

    # Optimized count_any using pre-defined lists
    def count_any(words: list[str]) -> int:
        return sum(1 for w in words if w in p)

    # Or even better, if we pre-compile keywords into a regex?
    # Actually, the original implementation iterates over a list.
    # Keeping the logic same but moving lists to module level avoids list creation overhead.

    correctness_hits = count_any(_RE_CORRECTNESS_KEYWORDS)
    if _RE_CONSTRAINTS.search(p):
        correctness_hits += 1
    correctness = min(1.0, correctness_hits / 6.0)

    quality_hits = count_any(_RE_QUALITY_KEYWORDS)
    if any(w in p for w in ["class ", "api", "library", "module"]):
        quality_hits += 1
    quality = min(1.0, quality_hits / 6.0)

    eff_hits = count_any(_RE_EFFICIENCY_KEYWORDS)
    if _RE_TIME_UNITS.search(p):
        eff_hits += 1
    if _RE_SIZE_UNITS.search(p):
        eff_hits += 1
    efficiency = min(1.0, eff_hits / 6.0)

    doc_hits = count_any(_RE_DOCS_KEYWORDS)
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

def _strip_code_fences_optimized(text: str) -> str:
    t = (text or "").strip()
    t = _RE_FENCE_START.sub("", t)
    t = _RE_FENCE_END.sub("", t)
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

    m = _RE_PERCENT.search(cleaned)
    if m:
        try:
            v = float(m.group(1)) / 100.0
            if 0.0 <= v <= 1.0:
                return v
        except Exception:
            pass

    m = _RE_FLOAT.search(cleaned)
    if m:
        try:
            v = float(m.group(0))
            if 0.0 <= v <= 1.0:
                return v
        except Exception:
            pass
    return None


# --- Benchmarking ---

def run_benchmark():
    iterations = 50000
    prompt = "Write a highly efficient, thread-safe python function to process 1000 items with documentation."
    teacher_response = "```python\n0.85\n```"

    print(f"Running benchmark with {iterations} iterations...")

    # Baseline: Rubric
    start = time.time()
    for _ in range(iterations):
        _rubric_difficulty_components_original(prompt, "python")
    end = time.time()
    rubric_baseline = end - start
    print(f"Original Rubric Difficulty: {rubric_baseline:.4f}s")

    # Optimized: Rubric
    start = time.time()
    for _ in range(iterations):
        _rubric_difficulty_components_optimized(prompt, "python")
    end = time.time()
    rubric_optimized = end - start
    print(f"Optimized Rubric Difficulty: {rubric_optimized:.4f}s")
    print(f"Speedup: {rubric_baseline / rubric_optimized:.2f}x")

    print("-" * 20)

    # Baseline: Extract Score
    start = time.time()
    for _ in range(iterations):
        _extract_score_original(teacher_response)
    end = time.time()
    extract_baseline = end - start
    print(f"Original Extract Score: {extract_baseline:.4f}s")

    # Optimized: Extract Score
    start = time.time()
    for _ in range(iterations):
        _extract_score_optimized(teacher_response)
    end = time.time()
    extract_optimized = end - start
    print(f"Optimized Extract Score: {extract_optimized:.4f}s")
    print(f"Speedup: {extract_baseline / extract_optimized:.2f}x")

if __name__ == "__main__":
    run_benchmark()
