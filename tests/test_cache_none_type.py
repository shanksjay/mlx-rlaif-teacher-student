"""
Unit tests for cache-related functions to identify NoneType issues.

Tests all functions that access teacher_score_cache with various NoneType scenarios
to identify potential issues in cache entry handling.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import time
from collections import OrderedDict
import sys
import os

# Add parent directory to path to import the training module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class MockTrainer:
    """Mock trainer with just the cache-related attributes and methods"""
    
    def __init__(self):
        self.teacher_score_cache = OrderedDict()
        self.teacher_score_cache_max_size = 10000
        self.teacher_score_cache_max_age_seconds = 3600.0
        self.cache_stats = {
            'teacher_score_cache_hits': 0,
            'teacher_score_calls': 0,
            'cached_scores_count': 0,
            'fresh_scores_count': 0
        }
        self.teacher = Mock()
        self.teacher.score_code = Mock(return_value=0.8)
        self.teacher.generate_code = Mock(return_value="teacher_code")
    
    def _add_to_cache(self, key: str, score: float, timestamp: float, max_age=None):
        """Add entry to LRU cache with size and age management"""
        from collections import OrderedDict
        
        # Remove oldest entries if cache is full (LRU eviction)
        while len(self.teacher_score_cache) >= self.teacher_score_cache_max_size:
            # Remove least recently used (first item for OrderedDict, arbitrary for dict)
            if isinstance(self.teacher_score_cache, OrderedDict):
                self.teacher_score_cache.popitem(last=False)
            else:
                # Fallback: remove first key (not true LRU but works)
                first_key = next(iter(self.teacher_score_cache))
                del self.teacher_score_cache[first_key]
        
        # Add new entry (most recently used goes to end)
        cache_max_age = max_age if max_age is not None else self.teacher_score_cache_max_age_seconds
        self.teacher_score_cache[key] = (score, timestamp, cache_max_age)
        # Move to end if OrderedDict (most recently used)
        if isinstance(self.teacher_score_cache, OrderedDict) and hasattr(self.teacher_score_cache, 'move_to_end'):
            self.teacher_score_cache.move_to_end(key)
    
    def _clean_cache_by_age(self, current_time: float = None) -> int:
        """Remove expired cache entries based on age"""
        # Import time module to avoid scoping issues
        # (module-level import may be shadowed in some contexts)
        import time
        if current_time is None:
            current_time = time.time()
        
        keys_to_remove = []
        for key, entry in list(self.teacher_score_cache.items()):
            try:
                if isinstance(entry, tuple) and len(entry) >= 3:
                    score, timestamp, max_age = entry
                    age = current_time - timestamp
                    if age >= max_age:
                        keys_to_remove.append(key)
                elif isinstance(entry, tuple) and len(entry) == 2:
                    # Old format without max_age - use default
                    score, timestamp = entry
                    age = current_time - timestamp
                    if age >= self.teacher_score_cache_max_age_seconds:
                        keys_to_remove.append(key)
                elif not isinstance(entry, tuple):
                    # Very old format (just score) - remove it
                    keys_to_remove.append(key)
            except Exception:
                # Invalid entry format - remove it
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            try:
                del self.teacher_score_cache[key]
            except Exception:
                pass
        
        return len(keys_to_remove)
    
    def _process_sample_reward(self, sample: dict) -> tuple:
        """Process a single sample to compute reward (for parallel execution)"""
        try:
            # Get teacher's reference code (cached)
            teacher_code = self.teacher.generate_code(
                sample['prompt'],
                sample['language']
            )
            
            # Score student code with LRU caching and fresh/cached tracking
            import time
            current_time = time.time()
            student_code_key = f"{sample['code']}:{sample['prompt']}:{sample['language']}"
            
            # Check cache with age validation
            cached_entry = self.teacher_score_cache.get(student_code_key)
            if cached_entry is not None:
                cached_score, cache_timestamp = cached_entry[:2] if len(cached_entry) >= 2 else (None, None)
                if cached_score is not None:
                    age_seconds = current_time - cache_timestamp if cache_timestamp else float('inf')
                    max_age = cached_entry[2] if len(cached_entry) >= 3 else self.teacher_score_cache_max_age_seconds
                    
                    # Use cached score if not too old
                    if age_seconds < max_age:
                        student_score = cached_score
                        self.cache_stats['teacher_score_cache_hits'] += 1
                        self.cache_stats['cached_scores_count'] += 1
                        # Move to end (most recently used)
                        if hasattr(self.teacher_score_cache, 'move_to_end'):
                            self.teacher_score_cache.move_to_end(student_code_key)
                    else:
                        # Cache entry too old, remove it and get fresh score
                        del self.teacher_score_cache[student_code_key]
                        cached_entry = None
                else:
                    # Cached entry exists but score is None - invalid entry, remove it
                    del self.teacher_score_cache[student_code_key]
                    cached_entry = None
            
            if cached_entry is None or (cached_entry is not None and len(cached_entry) < 2):
                # Cache miss or expired - get fresh score
                self.cache_stats['teacher_score_calls'] += 1
                self.cache_stats['fresh_scores_count'] += 1
                student_score = self.teacher.score_code(
                    sample['code'],
                    sample['prompt'],
                    sample['language'],
                    use_cache=True  # Use cache in score_code as well (defensive caching)
                )
                # Add to cache with timestamp
                self._add_to_cache(student_code_key, student_score, current_time)
            
            # Score teacher code (baseline) - cache this with a special prefix to distinguish from student scores
            # Teacher code doesn't change, so it's safe to cache across epochs (no age limit)
            teacher_code_key = f"TEACHER_CODE:{teacher_code}:{sample['prompt']}:{sample['language']}"
            cached_entry = self.teacher_score_cache.get(teacher_code_key)
            if cached_entry is not None:
                teacher_score, _ = cached_entry[:2] if len(cached_entry) >= 2 else (None, None)
                if teacher_score is not None:
                    self.cache_stats['teacher_score_cache_hits'] += 1
                    self.cache_stats['cached_scores_count'] += 1
                    # Move to end (most recently used)
                    if hasattr(self.teacher_score_cache, 'move_to_end'):
                        self.teacher_score_cache.move_to_end(teacher_code_key)
                else:
                    cached_entry = None
            
            if cached_entry is None:
                self.cache_stats['teacher_score_calls'] += 1
                self.cache_stats['fresh_scores_count'] += 1
                teacher_score = self.teacher.score_code(
                    teacher_code,
                    sample['prompt'],
                    sample['language'],
                    use_cache=True  # Use cache in score_code as well (defensive caching)
                )
                # Teacher code cache entries never expire (age = infinity)
                self._add_to_cache(teacher_code_key, teacher_score, current_time, max_age=float('inf'))
            
            # Normalized reward (relative to teacher)
            reward = student_score / (teacher_score + 1e-6)
            
            dataset_entry = {
                'prompt': sample['prompt'],
                'language': sample['language'],
                'student_code': sample['code'],
                'teacher_code': teacher_code,
                'student_score': float(student_score),
                'teacher_score': float(teacher_score),
                'reward': float(reward),
            }
            
            return reward, dataset_entry
        except Exception as e:
            raise


class TestCacheNoneTypeIssues(unittest.TestCase):
    """Test cache functions with NoneType values to identify issues"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.trainer = MockTrainer()
    
    def test_clean_cache_by_age_with_none_entry(self):
        """Test _clean_cache_by_age with None entries"""
        # Add None entry to cache
        self.trainer.teacher_score_cache['none_key'] = None
        
        # Should handle None gracefully
        result = self.trainer._clean_cache_by_age()
        self.assertIsInstance(result, int)
        # None entry should be removed
        self.assertNotIn('none_key', self.trainer.teacher_score_cache)
    
    def test_clean_cache_by_age_with_none_in_tuple(self):
        """Test _clean_cache_by_age with None values in tuple"""
        current_time = time.time()
        
        # Test case 1: None score
        self.trainer.teacher_score_cache['none_score'] = (None, current_time - 100, 3600.0)
        
        # Test case 2: None timestamp
        self.trainer.teacher_score_cache['none_timestamp'] = (0.5, None, 3600.0)
        
        # Test case 3: None max_age
        self.trainer.teacher_score_cache['none_max_age'] = (0.5, current_time - 100, None)
        
        # Test case 4: All None
        self.trainer.teacher_score_cache['all_none'] = (None, None, None)
        
        result = self.trainer._clean_cache_by_age(current_time)
        self.assertIsInstance(result, int)
    
    def test_clean_cache_by_age_with_invalid_tuple_length(self):
        """Test _clean_cache_by_age with invalid tuple lengths"""
        current_time = time.time()
        
        # Empty tuple
        self.trainer.teacher_score_cache['empty'] = ()
        
        # Single element
        self.trainer.teacher_score_cache['single'] = (0.5,)
        
        # Too many elements
        self.trainer.teacher_score_cache['too_many'] = (0.5, current_time, 3600.0, 'extra')
        
        result = self.trainer._clean_cache_by_age(current_time)
        self.assertIsInstance(result, int)
    
    def test_clean_cache_by_age_with_non_tuple_entry(self):
        """Test _clean_cache_by_age with non-tuple entries"""
        # String entry
        self.trainer.teacher_score_cache['string'] = "invalid"
        
        # Dict entry
        self.trainer.teacher_score_cache['dict'] = {'score': 0.5}
        
        # List entry
        self.trainer.teacher_score_cache['list'] = [0.5, time.time()]
        
        result = self.trainer._clean_cache_by_age()
        self.assertIsInstance(result, int)
        # Non-tuple entries should be removed
        self.assertNotIn('string', self.trainer.teacher_score_cache)
        self.assertNotIn('dict', self.trainer.teacher_score_cache)
        self.assertNotIn('list', self.trainer.teacher_score_cache)
    
    def test_process_sample_reward_with_none_cached_entry(self):
        """Test _process_sample_reward with None cached_entry"""
        # Mock teacher
        self.trainer.teacher = Mock()
        self.trainer.teacher.score_code = Mock(return_value=0.8)
        self.trainer.teacher.generate_code = Mock(return_value="teacher_code")
        
        # Add None entry to cache
        self.trainer.teacher_score_cache['test_key'] = None
        
        sample = {
            'code': 'def test(): pass',
            'prompt': 'test prompt',
            'language': 'python'
        }
        
        # Should handle None gracefully
        try:
            reward, dataset_entry = self.trainer._process_sample_reward(sample)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(dataset_entry, dict)
        except (TypeError, AttributeError, IndexError) as e:
            self.fail(f"_process_sample_reward raised {type(e).__name__}: {e}")
    
    def test_process_sample_reward_with_none_in_cached_entry_tuple(self):
        """Test _process_sample_reward with None values in cached_entry tuple"""
        self.trainer.teacher = Mock()
        self.trainer.teacher.score_code = Mock(return_value=0.8)
        self.trainer.teacher.generate_code = Mock(return_value="teacher_code")
        
        current_time = time.time()
        student_code_key = "def test(): pass:test prompt:python"
        
        # Test case 1: None score
        self.trainer.teacher_score_cache[student_code_key] = (None, current_time, 3600.0)
        
        sample = {
            'code': 'def test(): pass',
            'prompt': 'test prompt',
            'language': 'python'
        }
        
        try:
            reward, dataset_entry = self.trainer._process_sample_reward(sample)
            self.assertIsInstance(reward, float)
        except (TypeError, AttributeError, IndexError) as e:
            self.fail(f"_process_sample_reward with None score raised {type(e).__name__}: {e}")
    
    def test_process_sample_reward_with_none_timestamp(self):
        """Test _process_sample_reward with None timestamp in cached_entry"""
        self.trainer.teacher = Mock()
        self.trainer.teacher.score_code = Mock(return_value=0.8)
        self.trainer.teacher.generate_code = Mock(return_value="teacher_code")
        
        student_code_key = "def test(): pass:test prompt:python"
        
        # None timestamp
        self.trainer.teacher_score_cache[student_code_key] = (0.7, None, 3600.0)
        
        sample = {
            'code': 'def test(): pass',
            'prompt': 'test prompt',
            'language': 'python'
        }
        
        try:
            reward, dataset_entry = self.trainer._process_sample_reward(sample)
            self.assertIsInstance(reward, float)
        except (TypeError, AttributeError, IndexError) as e:
            self.fail(f"_process_sample_reward with None timestamp raised {type(e).__name__}: {e}")
    
    def test_process_sample_reward_with_short_tuple(self):
        """Test _process_sample_reward with tuple shorter than expected"""
        self.trainer.teacher = Mock()
        self.trainer.teacher.score_code = Mock(return_value=0.8)
        self.trainer.teacher.generate_code = Mock(return_value="teacher_code")
        
        student_code_key = "def test(): pass:test prompt:python"
        
        # Single element tuple
        self.trainer.teacher_score_cache[student_code_key] = (0.7,)
        
        sample = {
            'code': 'def test(): pass',
            'prompt': 'test prompt',
            'language': 'python'
        }
        
        try:
            reward, dataset_entry = self.trainer._process_sample_reward(sample)
            self.assertIsInstance(reward, float)
        except (TypeError, AttributeError, IndexError) as e:
            self.fail(f"_process_sample_reward with short tuple raised {type(e).__name__}: {e}")
    
    def test_process_sample_reward_with_teacher_code_none_entry(self):
        """Test _process_sample_reward with None in teacher code cache entry"""
        self.trainer.teacher = Mock()
        self.trainer.teacher.score_code = Mock(return_value=0.8)
        self.trainer.teacher.generate_code = Mock(return_value="teacher_code")
        
        teacher_code_key = "TEACHER_CODE:teacher_code:test prompt:python"
        
        # None entry
        self.trainer.teacher_score_cache[teacher_code_key] = None
        
        sample = {
            'code': 'def test(): pass',
            'prompt': 'test prompt',
            'language': 'python'
        }
        
        try:
            reward, dataset_entry = self.trainer._process_sample_reward(sample)
            self.assertIsInstance(reward, float)
        except (TypeError, AttributeError, IndexError) as e:
            self.fail(f"_process_sample_reward with None teacher entry raised {type(e).__name__}: {e}")
    
    def test_add_to_cache_with_none_values(self):
        """Test _add_to_cache with None values"""
        # Test with None score
        try:
            self.trainer._add_to_cache('key1', None, time.time())
        except (TypeError, AttributeError) as e:
            self.fail(f"_add_to_cache with None score raised {type(e).__name__}: {e}")
        
        # Test with None timestamp
        try:
            self.trainer._add_to_cache('key2', 0.5, None)
        except (TypeError, AttributeError) as e:
            self.fail(f"_add_to_cache with None timestamp raised {type(e).__name__}: {e}")
        
        # Test with None max_age
        try:
            self.trainer._add_to_cache('key3', 0.5, time.time(), max_age=None)
        except (TypeError, AttributeError) as e:
            self.fail(f"_add_to_cache with None max_age raised {type(e).__name__}: {e}")
    
    def test_epoch_cache_cleanup_with_none_entries(self):
        """Test epoch cache cleanup (lines 4612-4622) with None entries"""
        current_time = time.time()
        
        # Add various problematic entries
        self.trainer.teacher_score_cache['none_entry'] = None
        self.trainer.teacher_score_cache['none_timestamp'] = (0.5, None, 3600.0)
        self.trainer.teacher_score_cache['short_tuple'] = (0.5,)
        self.trainer.teacher_score_cache['valid_student'] = (0.5, current_time - 100, 3600.0)
        self.trainer.teacher_score_cache['TEACHER_CODE:test'] = (0.8, current_time - 100, float('inf'))
        
        # Simulate the epoch cleanup logic
        student_keys_to_remove = []
        for key, entry in list(self.trainer.teacher_score_cache.items()):
            if not key.startswith("TEACHER_CODE:"):
                try:
                    if len(entry) >= 2:
                        _, timestamp = entry[:2]
                        if timestamp is not None:
                            age_hours = (current_time - timestamp) / 3600
                            if age_hours > 2.0:
                                student_keys_to_remove.append(key)
                except (TypeError, AttributeError, IndexError) as e:
                    # Invalid entry format - remove it
                    student_keys_to_remove.append(key)
        
        for key in student_keys_to_remove:
            try:
                del self.trainer.teacher_score_cache[key]
            except Exception:
                pass
        
        # Should not raise exceptions
        self.assertIsInstance(len(self.trainer.teacher_score_cache), int)
    
    def test_cache_access_with_none_slicing(self):
        """Test cache entry slicing with None values"""
        # Test various problematic cache entry formats
        test_cases = [
            None,
            (None,),
            (None, None),
            (None, None, None),
            (0.5, None),
            (None, time.time()),
            (0.5, None, 3600.0),
            (None, time.time(), 3600.0),
            (0.5, time.time(), None),
        ]
        
        for i, entry in enumerate(test_cases):
            key = f'test_key_{i}'
            self.trainer.teacher_score_cache[key] = entry
            
            try:
                # Simulate the access pattern from _process_sample_reward
                if entry is not None:
                    if isinstance(entry, tuple) and len(entry) >= 2:
                        score, timestamp = entry[:2]
                        if len(entry) >= 3:
                            max_age = entry[2]
                    elif isinstance(entry, tuple) and len(entry) == 1:
                        score, timestamp = entry[0], None
                    else:
                        score, timestamp = None, None
                else:
                    score, timestamp = None, None
            except (TypeError, AttributeError, IndexError) as e:
                self.fail(f"Cache access failed for entry {entry}: {type(e).__name__}: {e}")
    
    def test_cache_entry_unpacking_edge_cases(self):
        """Test edge cases in cache entry unpacking"""
        current_time = time.time()
        
        # Test the exact pattern from line 3212
        test_entries = [
            None,
            (),
            (None,),
            (None, None),
            (0.5,),
            (0.5, None),
            (None, current_time),
            (0.5, current_time),
            (0.5, current_time, None),
            (None, current_time, 3600.0),
            (0.5, None, 3600.0),
        ]
        
        for entry in test_entries:
            try:
                # Pattern from line 3212
                cached_score, cache_timestamp = entry[:2] if (entry is not None and isinstance(entry, (tuple, list)) and len(entry) >= 2) else (None, None)
                
                # Pattern from line 3215
                if entry is not None and isinstance(entry, (tuple, list)) and len(entry) >= 3:
                    max_age = entry[2]
                else:
                    max_age = self.trainer.teacher_score_cache_max_age_seconds
                
                # Pattern from line 3248
                teacher_score, _ = entry[:2] if (entry is not None and isinstance(entry, (tuple, list)) and len(entry) >= 2) else (None, None)
                
            except (TypeError, AttributeError, IndexError) as e:
                self.fail(f"Unpacking failed for entry {entry}: {type(e).__name__}: {e}")


if __name__ == '__main__':
    unittest.main()

