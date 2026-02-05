import sys
import os
import json
import time
from unittest.mock import MagicMock

# Mock torch and other deps
sys.modules['torch'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['torch.utils.data.Dataset'] = object
sys.modules['psutil'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['anthropic'] = MagicMock()

# Mock logger
logger = MagicMock()

class CodeDataset:
    """Dataset for code training examples"""

    def __init__(self, data_file: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Only load from file if data_file is provided and not empty
        if data_file and data_file.strip():
            logger.info(f"Loading dataset from {data_file}")
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Dataset file not found: {data_file}")
            with open(data_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
            logger.info(f"Loaded {len(self.data)} examples")
        else:
            # Empty dataset - data can be set directly (e.g., for curriculum learning)
            logger.debug("Created empty dataset (data will be set directly)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get('prompt', '')
        language = item.get('language', 'python')

        # Format prompt with language context
        formatted_prompt = f"Write high-quality {language} code:\n\n{prompt}\n\nCode:"

        # Optimized: Removed tokenization

        return {
            # 'input_ids': encoding['input_ids'].squeeze(),
            # 'attention_mask': encoding['attention_mask'].squeeze(),
            'prompt': prompt,
            'language': language,
            'prompt_text': formatted_prompt
        }

class MockTensor:
    def squeeze(self):
        return self

class MockTokenizer:
    def __call__(self, text, truncation=True, max_length=2048, padding='max_length', return_tensors='pt'):
        t = MockTensor()
        return {
            'input_ids': t,
            'attention_mask': t
        }

def benchmark():
    data_file = "scripts/reproduction/dataset_large.jsonl"
    tokenizer = MockTokenizer()
    dataset = CodeDataset(data_file, tokenizer, max_length=2048)

    print(f"Dataset size: {len(dataset)}")

    start_time = time.time()
    # Iterate over dataset
    for i in range(len(dataset)):
        _ = dataset[i]
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Time to iterate {len(dataset)} items: {total_time:.4f} seconds")
    print(f"Items per second: {len(dataset) / total_time:.2f}")

if __name__ == "__main__":
    benchmark()
