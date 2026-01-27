
import sys
import os
import time
import json
import logging
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MOCKING ---
# We must mock all external dependencies before importing train_rlaif

# Mock modules
sys.modules['yaml'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['anthropic'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['transformers'] = MagicMock()

# Mock torch and submodules
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.tensorboard'] = MagicMock()

# Dataset needs to be a class we can inherit from (or mock properly)
# Since CodeDataset inherits from Dataset, we need torch.utils.data.Dataset to be a class
class MockDataset:
    pass
mock_data = MagicMock()
mock_data.Dataset = MockDataset
# Mock DataLoader too
class MockDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, collate_fn=None, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def __iter__(self):
        batch = []
        for i in range(len(self.dataset)):
            batch.append(self.dataset[i])
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch:
            yield self._collate(batch)

    def _collate(self, batch):
        elem = batch[0]
        if isinstance(elem, dict):
            return {key: [d[key] for d in batch] for key in elem}
        return batch

mock_data.DataLoader = MockDataLoader
sys.modules['torch.utils.data'] = mock_data

# Ensure numpy import works as 'import numpy as np'
sys.modules['numpy'].__name__ = 'numpy'

# --- END MOCKING ---

# Add scripts/training to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../training'))

# Import the module
try:
    import train_rlaif
    from train_rlaif import CodeDataset
except ImportError as e:
    print(f"Failed to import train_rlaif: {e}")
    sys.exit(1)

def create_dummy_data(filename, num_samples=1000):
    with open(filename, 'w') as f:
        for i in range(num_samples):
            item = {
                "prompt": f"Write a function to calculate fibonacci number {i}",
                "language": "python"
            }
            f.write(json.dumps(item) + "\n")

def verify_optimization():
    data_file = "verify_data.jsonl"
    num_samples = 1000
    create_dummy_data(data_file, num_samples=num_samples)

    # Mock tokenizer
    tokenizer = MagicMock()
    # If tokenizer is called, it returns a dict. We want to ensure it is NOT called.
    tokenizer.return_value = {'input_ids': MagicMock(), 'attention_mask': MagicMock()}

    print("Initializing CodeDataset from train_rlaif.py...")
    try:
        dataset = CodeDataset(data_file, tokenizer, max_length=2048)
    except Exception as e:
        print(f"Failed to instantiate CodeDataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"Dataset size: {len(dataset)}")

    print("Iterating over dataset...")
    start_time = time.time()

    # We use our MockDataLoader to iterate
    loader = mock_data.DataLoader(dataset, batch_size=32, shuffle=False)

    for batch in loader:
        # Verify structure
        if 'input_ids' in batch:
            print("FAILURE: 'input_ids' found in batch!")
            sys.exit(1)
        if 'prompt' not in batch:
            print("FAILURE: 'prompt' missing from batch!")
            sys.exit(1)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total time: {total_time:.4f}s")
    print(f"Samples per second: {num_samples / total_time:.2f}")

    # Verify tokenizer was NOT called
    if tokenizer.call_count > 0:
        print(f"FAILURE: Tokenizer was called {tokenizer.call_count} times!")
        sys.exit(1)
    else:
        print("SUCCESS: Tokenizer was NOT called.")

    # Clean up
    if os.path.exists(data_file):
        os.remove(data_file)

if __name__ == "__main__":
    verify_optimization()
