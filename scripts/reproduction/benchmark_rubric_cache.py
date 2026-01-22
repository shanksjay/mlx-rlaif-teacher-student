
import time
import sys
import os
import random
from functools import lru_cache

# Add scripts/training to path to import train_rlaif
sys.path.append(os.path.abspath("scripts/training"))

# Import the function and necessary globals if possible,
# but easiest is to import the module and access the function
import train_rlaif

# Create a cached version for comparison
# We can't easily decorate the function in the module after import for the benchmark
# without modifying the module, but we can wrap it.
# However, the goal is to measure the benefit if we applied it inside the module.

def benchmark():
    print("Benchmarking _rubric_difficulty_components...")

    # Generate some dummy prompts
    prompts = [
        "Write a python function to sort a list efficiently.",
        "Create a thread-safe singleton class in C++.",
        "Implement a binary search tree in Rust with proper error handling.",
        "Write a function to validate email addresses using regex.",
        "Explain how to use the 'requests' library in Python.",
        "Write a function to calculate fibonacci numbers recursively.",
        "Implement a merge sort algorithm.",
        "Create a REST API using Flask.",
        "Write a script to parse a CSV file and calculate statistics.",
        "Implement a producer-consumer problem using semaphores."
    ] * 100  # 1000 prompts

    languages = ["python", "cpp", "rust", "python", "python", "python", "cpp", "python", "python", "cpp"] * 100

    # 1. Measure baseline
    start_time = time.time()
    for p, l in zip(prompts, languages):
        _ = train_rlaif._rubric_difficulty_components(p, l)
    end_time = time.time()
    baseline_time = end_time - start_time
    print(f"Baseline time (1000 calls): {baseline_time:.4f} seconds")

    # 2. Measure with cache
    # We decorate the function from the module
    cached_func = lru_cache(maxsize=1024)(train_rlaif._rubric_difficulty_components)

    # Warmup (populate cache)
    # We simulate the training loop where we might call it multiple times for same prompt
    # Scenario: 4 samples per prompt.

    # Re-create the workload: 100 unique prompts, each called 4 times
    unique_prompts = prompts[:10]
    unique_languages = languages[:10]

    workload_prompts = []
    workload_languages = []
    for p, l in zip(unique_prompts, unique_languages):
        for _ in range(4): # 4 samples per prompt
            workload_prompts.append(p)
            workload_languages.append(l)

    # Expand to a larger workload to get measurable time
    workload_prompts *= 100
    workload_languages *= 100
    # Total 4000 calls, 1000 unique inputs? No, 10 unique inputs repeated.
    # Training usually iterates over dataset.
    # Let's say dataset has 1000 examples.
    # For each example, we generate 4 samples.
    # So 4 calls with same args.

    real_prompts = [f"Prompt {i} about code" for i in range(1000)]
    real_languages = ["python"] * 1000

    # Baseline on realistic workload
    start_time = time.time()
    for p, l in zip(real_prompts, real_languages):
        for _ in range(4):
            _ = train_rlaif._rubric_difficulty_components(p, l)
    end_time = time.time()
    baseline_real_time = end_time - start_time
    print(f"Baseline time (4000 calls, 4 calls/prompt): {baseline_real_time:.4f} seconds")

    # Cached on realistic workload
    start_time = time.time()
    for p, l in zip(real_prompts, real_languages):
        for _ in range(4):
            _ = cached_func(p, l)
    end_time = time.time()
    cached_real_time = end_time - start_time
    print(f"Cached time (4000 calls, 4 calls/prompt): {cached_real_time:.4f} seconds")

    speedup = baseline_real_time / cached_real_time
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark()
