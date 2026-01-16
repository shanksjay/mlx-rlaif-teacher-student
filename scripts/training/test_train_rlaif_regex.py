
import sys
import os
import unittest
from pathlib import Path

# Add repo root to path to allow importing scripts.training.train_rlaif
sys.path.append(str(Path(__file__).resolve().parents[2]))

# We need to mock some imports or environment variables if the script does heavy initialization on import
# But let's try importing directly first.
try:
    from scripts.training.train_rlaif import _rubric_difficulty_components
except ImportError as e:
    print(f"Import failed: {e}")
    # If import fails (e.g. due to missing deps in this environment), we might need to mock
    _rubric_difficulty_components = None

# These will be available after refactoring
try:
    from scripts.training.train_rlaif import _strip_code_fences, _extract_score
except ImportError:
    _strip_code_fences = None
    _extract_score = None

class TestRegexOptimizations(unittest.TestCase):
    def test_rubric_difficulty_components(self):
        if _rubric_difficulty_components is None:
            self.skipTest("Could not import _rubric_difficulty_components")

        # Test 1: Constraints (should trigger correctness hits)
        # "must" is a trigger word
        res = _rubric_difficulty_components("This must be efficient", "python")
        # Check that it runs without error and returns a dict
        self.assertIsInstance(res, dict)
        self.assertIn("rubric_demand", res)

        # Test 2: Time/Size units (should trigger efficiency hits)
        # "1000 items", "5ms" are triggers
        res_perf = _rubric_difficulty_components("Process 1000 items in 5ms", "cpp")
        self.assertGreater(res_perf["efficiency"], 0)

    def test_strip_code_fences(self):
        if _strip_code_fences is None:
            print("Skipping test_strip_code_fences (function not yet exposed)")
            return

        self.assertEqual(_strip_code_fences("```python\nprint('hi')\n```"), "print('hi')")
        self.assertEqual(_strip_code_fences("```\ncode\n```"), "code")
        self.assertEqual(_strip_code_fences("no fences"), "no fences")
        self.assertEqual(_strip_code_fences("   ```rust\nfn main() {}\n```   "), "fn main() {}")

    def test_extract_score(self):
        if _extract_score is None:
            print("Skipping test_extract_score (function not yet exposed)")
            return

        self.assertEqual(_extract_score("0.95"), 0.95)
        self.assertEqual(_extract_score("Score: 0.8"), 0.8)
        self.assertEqual(_extract_score("95%"), 0.95)
        self.assertEqual(_extract_score("```\n0.7\n```"), 0.7)
        self.assertEqual(_extract_score("The score is 0.5."), 0.5)
        self.assertIsNone(_extract_score(""))
        self.assertIsNone(_extract_score("No score here"))

if __name__ == "__main__":
    unittest.main()
