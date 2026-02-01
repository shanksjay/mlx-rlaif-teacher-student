
import sys
import os
import time
import json
from unittest.mock import MagicMock

# 1. Setup Mocks BEFORE importing train_rlaif
# Mock torch
mock_torch = MagicMock()
mock_torch.utils.data.Dataset = object
mock_torch.utils.data.DataLoader = MagicMock()
# Mock tensor attributes that might be accessed
mock_torch.Tensor = MagicMock
mock_torch.long = MagicMock()

sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.utils"] = MagicMock()
sys.modules["torch.utils.data"] = mock_torch.utils.data
sys.modules["torch.utils.tensorboard"] = MagicMock()

# Mock other deps
sys.modules["transformers"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["anthropic"] = MagicMock()
sys.modules["psutil"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["tqdm"] = MagicMock()
sys.modules["yaml"] = MagicMock()  # Added yaml

# 2. Import CodeDataset
sys.path.append(os.path.join(os.getcwd(), 'scripts/training'))

try:
    from train_rlaif import CodeDataset
except ImportError as e:
    print(f"Failed to import CodeDataset: {e}")
    sys.exit(1)

def benchmark_dataset():
    # Create a dummy dataset file
    data = [{"prompt": f"Write a function to do X_{i}", "language": "python"} for i in range(1000)]
    data_file = "dummy_data.jsonl"
    with open(data_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # Mock tokenizer
    tokenizer = MagicMock()
    # Simulate return value
    # We return a simple dict.
    # Important: The values in dict should support .squeeze()
    mock_tensor = MagicMock()
    mock_tensor.squeeze.return_value = "tensor"

    tokenizer.return_value = {
        'input_ids': mock_tensor,
        'attention_mask': mock_tensor
    }

    print("Initializing CodeDataset...")
    dataset = CodeDataset(data_file, tokenizer, max_length=2048)

    print("Benchmarking iteration speed...")
    start_time = time.time()
    for i in range(len(dataset)):
        _ = dataset[i]
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / len(dataset)
    print(f"Total time for {len(dataset)} items: {total_time:.4f}s")
    print(f"Average time per item: {avg_time:.6f}s")
    print(f"Items per second: {len(dataset)/total_time:.2f}")

    # Verify tokenizer call count
    print(f"Tokenizer call count: {tokenizer.call_count}")

    if os.path.exists(data_file):
        os.remove(data_file)

if __name__ == "__main__":
    benchmark_dataset()
