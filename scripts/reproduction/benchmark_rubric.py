import time
import sys
import os

# Ensure scripts can be imported
sys.path.append(os.getcwd())

from scripts.training.train_rlaif import _rubric_difficulty_components, _extract_score

def benchmark_rubric():
    print("Benchmarking _rubric_difficulty_components...")
    prompts = [
        "Write a python function to sort a list using quicksort. Handle edge cases like empty list. Make it efficient O(n log n).",
        "Create a thread-safe singleton class in C++ with RAII.",
        "Simple hello world in Rust.",
        "Implement a JSON parser that handles invalid input and unicode characters.",
        "Write a function to calculate fibonacci sequence. optimize for time complexity.",
    ] * 100 # 500 prompts

    start = time.time()
    # Run enough times to get stable measurement
    iterations = 200
    for _ in range(iterations):
        for p in prompts:
            _rubric_difficulty_components(p, "python")
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")

def benchmark_extract_score():
    print("Benchmarking _extract_score...")
    texts = [
        "0.75",
        "The score is 0.8.",
        "Score: 95%",
        "```\n0.5\n```",
        "I give this a 0.2 because...",
        "Perfect score 1.0",
    ] * 1000

    start = time.time()
    iterations = 200
    for _ in range(iterations):
        for t in texts:
            _extract_score(t)
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")

if __name__ == "__main__":
    benchmark_rubric()
    benchmark_extract_score()

def verify_correctness():
    print("Verifying correctness...")
    # Rubric check
    diff = _rubric_difficulty_components("Write a python function to sort a list using quicksort. Handle edge cases like empty list. Make it efficient O(n log n).", "python")
    print(f"Rubric: {diff}")
    assert diff['efficiency'] > 0
    assert diff['correctness'] > 0

    # Score check
    assert _extract_score("The score is 0.75") == 0.75
    assert _extract_score("Score: 95%") == 0.95
    assert _extract_score("```\n0.5\n```") == 0.5
    print("Correctness verified.")

if __name__ == "__main__":
    verify_correctness()
