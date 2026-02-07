
import sys
import time
import json
import os
from unittest.mock import MagicMock

# Mock heavy dependencies to avoid import errors and overhead
mock_torch = MagicMock()
mock_torch.utils = MagicMock()
mock_torch.utils.data = MagicMock()
# Make Dataset a class so CodeDataset can inherit from it
class MockDataset:
    pass
mock_torch.utils.data.Dataset = MockDataset

# Explicitly set attributes that might be accessed during import
mock_torch.Tensor = MagicMock()
mock_torch.nn = MagicMock()
mock_torch.backends = MagicMock()
mock_torch.backends.mps = MagicMock()
mock_torch.backends.mps.is_available.return_value = False

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch.nn
sys.modules["torch.utils"] = mock_torch.utils
sys.modules["torch.utils.data"] = mock_torch.utils.data
sys.modules["torch.utils.tensorboard"] = MagicMock()

sys.modules["transformers"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["anthropic"] = MagicMock()
sys.modules["psutil"] = MagicMock()
sys.modules["tensorboard"] = MagicMock()
sys.modules["yaml"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["tqdm"] = MagicMock()

# Determine the absolute path to scripts/training
current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.abspath(os.path.join(current_dir, "../training"))

# Add training directory to sys.path
if training_dir not in sys.path:
    sys.path.append(training_dir)

try:
    from train_rlaif import CodeDataset
except ImportError as e:
    print(f"Failed to import CodeDataset: {e}")
    # Print sys.path for debugging
    print(f"sys.path: {sys.path}")
    sys.exit(1)

def create_dummy_dataset(filename="dummy_data.jsonl", num_samples=1000):
    with open(filename, "w") as f:
        for i in range(num_samples):
            item = {
                "prompt": f"Write a function to calculate factorial of {i}",
                "language": "python"
            }
            f.write(json.dumps(item) + "\n")

def benchmark_dataset():
    data_file = "dummy_data.jsonl"
    if not os.path.exists(data_file):
        create_dummy_dataset(data_file)

    # Create a mock tokenizer
    mock_tokenizer = MagicMock()
    # Mock the return value of tokenizer call (encoding)
    mock_encoding = MagicMock()
    mock_encoding.__getitem__.return_value.squeeze.return_value = "mock_tensor"

    # Configure tokenizer to return a dict with squeeze() methods
    def tokenizer_side_effect(*args, **kwargs):
        return {
            'input_ids': MagicMock(squeeze=lambda: "ids"),
            'attention_mask': MagicMock(squeeze=lambda: "mask")
        }
    mock_tokenizer.side_effect = tokenizer_side_effect

    print("Initializing CodeDataset...")
    dataset = CodeDataset(data_file, mock_tokenizer, max_length=1024)

    # Initial run to warm up (and verify it works)
    print("Verifying initial access...")
    try:
        item = dataset[0]
        # We need to access input_ids to trigger tokenization if it's there
        if 'input_ids' in item:
            _ = item['input_ids']
        print(f"Sample item keys: {list(item.keys())}")
    except Exception as e:
        print(f"Error accessing dataset[0]: {e}")
        import traceback
        traceback.print_exc()
        return

    start_time = time.time()
    iterations = 10000
    print(f"Iterating {iterations} times...")

    for i in range(iterations):
        _ = dataset[i % len(dataset)]

    end_time = time.time()
    duration = end_time - start_time
    print(f"Time taken: {duration:.4f} seconds")
    print(f"Throughput: {iterations / duration:.2f} items/sec")

    # Cleanup
    if os.path.exists(data_file):
        os.remove(data_file)

if __name__ == "__main__":
    benchmark_dataset()
