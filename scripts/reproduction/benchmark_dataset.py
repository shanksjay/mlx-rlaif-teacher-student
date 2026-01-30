
import time
import json
import os
import sys
import types
from unittest.mock import MagicMock

# --- MOCKING INFRASTRUCTURE START ---
# We need to construct a fake torch package structure
torch_mock = types.ModuleType('torch')
torch_mock.nn = types.ModuleType('torch.nn')
torch_mock.utils = types.ModuleType('torch.utils')
torch_mock.utils.data = types.ModuleType('torch.utils.data')
torch_mock.utils.tensorboard = types.ModuleType('torch.utils.tensorboard')
torch_mock.backends = types.ModuleType('torch.backends')
torch_mock.backends.mps = types.ModuleType('torch.backends.mps')
torch_mock.cuda = types.ModuleType('torch.cuda')

# Populate sys.modules
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = torch_mock.nn
sys.modules['torch.utils'] = torch_mock.utils
sys.modules['torch.utils.data'] = torch_mock.utils.data
sys.modules['torch.utils.tensorboard'] = torch_mock.utils.tensorboard

# Mock other dependencies
sys.modules['transformers'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['anthropic'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
sys.modules['yaml'] = MagicMock()
sys.modules['mlx'] = MagicMock()
sys.modules['mlx.core'] = MagicMock()
sys.modules['mlx_lm'] = MagicMock()

# Mock Dataset and DataLoader
class MockDataset:
    def __getitem__(self, idx):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError

class MockDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.dataset):
            raise StopIteration

        batch = []
        for _ in range(self.batch_size):
            if self.idx >= len(self.dataset):
                break
            batch.append(self.dataset[self.idx])
            self.idx += 1

        if not batch:
            raise StopIteration

        # Collate (simplified)
        collated = {}
        for key in batch[0].keys():
            collated[key] = [item[key] for item in batch]
        return collated

# Attach Dataset/DataLoader to the mocked module
torch_mock.utils.data.Dataset = MockDataset
torch_mock.utils.data.DataLoader = MockDataLoader
class MockSampler:
    def __class_getitem__(cls, item):
        return cls
torch_mock.utils.data.Sampler = MockSampler
torch_mock.utils.tensorboard.SummaryWriter = MagicMock()
torch_mock.Tensor = MagicMock

# Add scripts/training to path
sys.path.append(os.path.join(os.getcwd(), 'scripts/training'))
# --- MOCKING INFRASTRUCTURE END ---

# Mock tokenizer
class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, truncation=True, max_length=512, padding='max_length', return_tensors='pt'):
        # Simulate tokenization work
        _ = [ord(c) for c in text]
        return {'input_ids': MagicMock(), 'attention_mask': MagicMock()}

    def __len__(self):
        return 1000

def create_dummy_data(filename, num_examples=1000):
    with open(filename, 'w') as f:
        for i in range(num_examples):
            json.dump({
                "prompt": f"Write a function to calculate fibonacci number {i}. " * 50,
                "language": "python"
            }, f)
            f.write('\n')

def benchmark():
    # Import CodeDataset from the real script
    from train_rlaif import CodeDataset

    data_file = "dummy_data.jsonl"
    create_dummy_data(data_file, num_examples=5000)

    tokenizer = MockTokenizer()

    print("Benchmarking CodeDataset (from train_rlaif.py)...")
    try:
        dataset = CodeDataset(data_file, tokenizer)
        loader = MockDataLoader(dataset, batch_size=32)

        start_time = time.time()
        for batch in loader:
            pass
        end_time = time.time()
        duration = end_time - start_time
        print(f"Time: {duration:.4f} seconds")
        print(f"Throughput: {5000 / duration:.2f} examples/sec")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    os.remove(data_file)

if __name__ == "__main__":
    benchmark()
