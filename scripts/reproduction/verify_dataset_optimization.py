
import sys
import os
from unittest.mock import MagicMock

# Mock dependencies
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['torch.utils.tensorboard'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['anthropic'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
sys.modules['yaml'] = MagicMock()

# Explicitly mock torch.utils.data.Dataset
class MockDataset:
    pass
sys.modules['torch.utils.data'].Dataset = MockDataset

# Fix path: scripts/reproduction -> ../training
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../training')))

# Import train_rlaif
try:
    import train_rlaif
except ImportError as e:
    print(f"Failed to import train_rlaif: {e}")
    # Print sys.path for debugging
    print(f"sys.path: {sys.path}")
    sys.exit(1)

def verify_code_dataset():
    print("Verifying CodeDataset optimization...")

    # Create dummy data file
    data_file = "dummy_verify.jsonl"
    import json
    with open(data_file, 'w') as f:
        f.write(json.dumps({"prompt": "test prompt", "language": "python"}) + "\n")

    try:
        # Mock tokenizer
        tokenizer = MagicMock()

        # Instantiate CodeDataset
        dataset = train_rlaif.CodeDataset(data_file, tokenizer)

        # Get item
        item = dataset[0]

        # Verify item content
        print(f"Item keys: {item.keys()}")

        if 'input_ids' in item:
            print("❌ input_ids found in dataset item (Optimization failed)")
            sys.exit(1)

        if 'attention_mask' in item:
            print("❌ attention_mask found in dataset item (Optimization failed)")
            sys.exit(1)

        if item['prompt'] != "test prompt":
            print("❌ Incorrect prompt")
            sys.exit(1)

        # Verify tokenizer was NOT called
        if tokenizer.call_count > 0 or tokenizer.called:
             # Wait, __init__ doesn't call tokenizer. __getitem__ should not call it.
             print("❌ Tokenizer was called! (Optimization failed)")
             sys.exit(1)

        print("✅ CodeDataset verified: No tokenization, correct keys.")

    finally:
        if os.path.exists(data_file):
            os.remove(data_file)

if __name__ == "__main__":
    verify_code_dataset()
