
import sys
import time
import json
import os
from unittest.mock import MagicMock

# Mock dependencies BEFORE importing train_rlaif
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.utils.data"] = MagicMock()
# We need Dataset to be a class we can inherit from if we want to be safe,
# but MagicMock usually works if we don't call super().__init__ or similar.
# train_rlaif.py: class CodeDataset(Dataset):
# So Dataset must be a type.
sys.modules["torch.utils.data"].Dataset = object

sys.modules["transformers"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["anthropic"] = MagicMock()
sys.modules["psutil"] = MagicMock()
sys.modules["torch.utils.tensorboard"] = MagicMock()
sys.modules["yaml"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["tqdm"] = MagicMock()

# Mock bitsandbytesConfig
sys.modules["transformers"].BitsAndBytesConfig = MagicMock

# Import the module
# We need to add scripts/training to path
sys.path.append(os.path.abspath("scripts/training"))

try:
    from train_rlaif import CodeDataset
except ImportError as e:
    print(f"Failed to import CodeDataset: {e}")
    sys.exit(1)

def test_performance():
    print("Testing CodeDataset performance (actual class)...")

    # Mock tokenizer
    tokenizer = MagicMock()
    # If the optimization is applied, tokenizer should NOT be called.
    # If it is called, we can count it.

    # Create a dummy data file
    data_file = "temp_test_data.jsonl"
    with open(data_file, "w") as f:
        for i in range(10000):
            f.write(json.dumps({"prompt": f"prompt {i}", "language": "python"}) + "\n")

    try:
        dataset = CodeDataset(data_file, tokenizer)

        start_time = time.time()
        for i in range(len(dataset)):
            _ = dataset[i]
        end_time = time.time()

        total_time = end_time - start_time
        print(f"Time to process {len(dataset)} items: {total_time:.4f}s")
        print(f"Items per second: {len(dataset)/total_time:.2f}")
        print(f"Tokenizer calls: {tokenizer.call_count}")

    finally:
        if os.path.exists(data_file):
            os.remove(data_file)

if __name__ == "__main__":
    test_performance()
