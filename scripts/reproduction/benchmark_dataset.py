
import time
import json
import os
import sys
from unittest.mock import MagicMock

# Mock torch and transformers to avoid installing them for this benchmark
sys.modules['torch'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['transformers'] = MagicMock()

# Mock Dataset class
class Dataset:
    pass
sys.modules['torch.utils.data'].Dataset = Dataset

# Import CodeDataset from train_rlaif.py (we need to load the file source and exec it because of imports)
# Actually, it's easier to just copy the class or import if I can mock everything.
# Let's try to import the file but mock dependencies.

import time
import tempfile

def benchmark():
    # Create a dummy dataset file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        for i in range(100):
            f.write(json.dumps({"prompt": f"Prompt {i}", "language": "python"}) + "\n")
        temp_path = f.name

    try:
        # Mock dependencies for train_rlaif
        sys.modules['torch'] = MagicMock()
        sys.modules['torch.nn'] = MagicMock()
        sys.modules['torch.utils.data'] = MagicMock()
        sys.modules['torch.utils.tensorboard'] = MagicMock()
        sys.modules['numpy'] = MagicMock()
        sys.modules['tqdm'] = MagicMock()
        sys.modules['psutil'] = MagicMock()
        sys.modules['transformers'] = MagicMock()
        sys.modules['datasets'] = MagicMock()
        sys.modules['openai'] = MagicMock()
        sys.modules['anthropic'] = MagicMock()

        # We need Dataset to be a class we can inherit from
        class MockDataset:
            pass
        sys.modules['torch.utils.data'].Dataset = MockDataset

        # We need to read the file content and extract CodeDataset class because of the heavy imports at top level
        # that might still fail or trigger things we don't want.
        # But let's try importing first, maybe mocks are enough.

        # Actually, reading file and exec specific part is safer.
        with open('scripts/training/train_rlaif.py', 'r') as f:
            content = f.read()

        # Extract CodeDataset class definition
        # It assumes CodeDataset starts with 'class CodeDataset' and ends before next class
        start = content.find('class CodeDataset')
        end = content.find('class BucketedCurriculumSampler')
        class_source = content[start:end]

        # We need logger
        import logging
        logging.basicConfig(level=logging.INFO)
        global logger
        logger = logging.getLogger(__name__)

        # Execute class definition
        exec(class_source, globals())

        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.return_value = {
            'input_ids': MagicMock(squeeze=lambda: MagicMock()),
            'attention_mask': MagicMock(squeeze=lambda: MagicMock())
        }

        # Instantiate
        dataset = CodeDataset(temp_path, tokenizer)

        # Benchmark iteration
        start_time = time.time()
        for i in range(len(dataset)):
            _ = dataset[i]
        end_time = time.time()

        print(f"Time to iterate {len(dataset)} items: {end_time - start_time:.4f}s")
        print(f"Tokenizer call count: {tokenizer.call_count}")

    finally:
        os.remove(temp_path)

if __name__ == "__main__":
    benchmark()
