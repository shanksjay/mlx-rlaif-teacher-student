
import timeit
import sys
import os
from unittest.mock import MagicMock

# Mock dependencies
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.utils"] = MagicMock()

class MockDataset:
    pass
sys.modules["torch.utils.data"] = MagicMock()
sys.modules["torch.utils.data"].Dataset = MockDataset
sys.modules["torch.utils.data"].DataLoader = MagicMock()

sys.modules["torch.utils.tensorboard"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["tqdm"] = MagicMock()
sys.modules["psutil"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["anthropic"] = MagicMock()

# Add scripts/training to python path
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../training'))
sys.path.append(path)

try:
    from train_rlaif import _rubric_difficulty_components, _extract_score
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def benchmark():
    prompt = "Write a python function to implement a thread-safe LRU cache that handles concurrent access and validates input. It should be efficient and well documented."
    language = "python"

    # Warmup
    for _ in range(100):
        _rubric_difficulty_components(prompt, language)

    # Benchmark rubric difficulty
    number = 50000
    time_rubric = timeit.timeit(lambda: _rubric_difficulty_components(prompt, language), number=number)
    print(f"Rubric Difficulty: Time for {number} calls: {time_rubric:.4f}s")
    print(f"Rubric Difficulty: Time per call: {time_rubric/number*1000:.4f}ms")

    # Benchmark extract score
    score_text = "0.75"
    number_score = 50000
    time_score = timeit.timeit(lambda: _extract_score(score_text), number=number_score)
    print(f"Extract Score: Time for {number_score} calls: {time_score:.4f}s")
    print(f"Extract Score: Time per call: {time_score/number_score*1000:.4f}ms")

if __name__ == "__main__":
    benchmark()
