
import unittest
import sys
import os
from collections import OrderedDict
from unittest.mock import MagicMock, patch

# Add scripts directory to path to allow importing train_rlaif
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# We need to mock many imports because train_rlaif imports heavy libraries at module level
# which might fail or be slow. However, since we installed dependencies, we might try importing directly.
# But RLAIFTrainer.__init__ does a lot of heavy lifting (loading models).
# We should subclass RLAIFTrainer and override __init__ to avoid loading models.

from training.train_rlaif import RLAIFTrainer, RLAIFConfig

class MockRLAIFTrainer(RLAIFTrainer):
    def __init__(self, config):
        # Skip super().__init__ to avoid loading models
        self.config = config
        self.teacher_score_cache = OrderedDict()
        self.teacher_score_cache_max_size = 10000
        self.teacher_score_cache_max_age_seconds = 3600
        # cache_stats is used in other methods but strictly not in _clean_cache_by_age (except for maybe logging?)
        # _clean_cache_by_age does not use cache_stats in the provided code.

class TestRLAIFTrainer(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock(spec=RLAIFConfig)
        self.trainer = MockRLAIFTrainer(self.config)

    def test_clean_cache_by_age_expired(self):
        current_time = 10000.0
        # Add expired item
        self.trainer.teacher_score_cache["expired"] = (0.5, current_time - 4000, 3600)
        # Add valid item
        self.trainer.teacher_score_cache["valid"] = (0.5, current_time - 100, 3600)

        removed = self.trainer._clean_cache_by_age(current_time)
        self.assertEqual(removed, 1)
        self.assertNotIn("expired", self.trainer.teacher_score_cache)
        self.assertIn("valid", self.trainer.teacher_score_cache)

    def test_clean_cache_by_age_limit(self):
        current_time = 10000.0
        # Add 1500 expired items
        for i in range(1500):
            self.trainer.teacher_score_cache[f"expired_{i}"] = (0.5, current_time - 4000, 3600)

        # Add 1 valid item
        self.trainer.teacher_score_cache["valid"] = (0.5, current_time - 100, 3600)

        # Should remove only 1000
        removed = self.trainer._clean_cache_by_age(current_time)
        self.assertEqual(removed, 1000)

        # First 1000 should be gone
        for i in range(1000):
            self.assertNotIn(f"expired_{i}", self.trainer.teacher_score_cache)

        # 1001-1499 should remain
        for i in range(1000, 1500):
            self.assertIn(f"expired_{i}", self.trainer.teacher_score_cache)

if __name__ == '__main__':
    unittest.main()
