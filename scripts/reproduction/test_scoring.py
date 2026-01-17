
import unittest
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

from train_rlaif import _rubric_difficulty_components

class TestRubricDifficulty(unittest.TestCase):
    def test_rubric_difficulty(self):
        # Test case 1: Complex prompt
        prompt = "Write a python function to implement a thread-safe LRU cache that handles concurrent access and validates input. It should be efficient and well documented."
        language = "python"

        diff = _rubric_difficulty_components(prompt, language)

        # Check if keys exist
        self.assertIn("correctness", diff)
        self.assertIn("code_quality", diff)
        self.assertIn("efficiency", diff)
        self.assertIn("documentation", diff)
        self.assertIn("rubric_demand", diff)

        # Expect non-zero scores
        self.assertGreater(diff["correctness"], 0)
        self.assertGreater(diff["efficiency"], 0)
        self.assertGreater(diff["documentation"], 0)

        # Test case 2: Simple prompt
        prompt_simple = "print hello world"
        diff_simple = _rubric_difficulty_components(prompt_simple, language)

        # Expect lower scores than complex prompt
        self.assertLess(diff_simple["correctness"], diff["correctness"])

    def test_keywords_detection(self):
        # Test specific keywords
        prompt = "thread-safe deadlock atomic"
        diff = _rubric_difficulty_components(prompt, "cpp")
        self.assertGreater(diff["correctness"], 0.4) # At least 3 hits / 6.0 = 0.5

        prompt = "clean readable design pattern"
        diff = _rubric_difficulty_components(prompt, "java")
        self.assertGreater(diff["code_quality"], 0.4)

if __name__ == "__main__":
    unittest.main()
