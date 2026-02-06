
import time
import json
import os
import sys
from unittest.mock import MagicMock

# Mock torch
sys.modules['torch'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()

# Mock Dataset class
class Dataset:
    pass

# Mock tokenizer
class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, truncation=True, max_length=2048, padding='max_length', return_tensors='pt', **kwargs):
        # Simulate tokenization workload (string operations + allocation)
        # Real tokenization is much slower, but this gives a lower bound
        tokens = [1] * max_length
        return {
            'input_ids': MagicMock(squeeze=lambda: tokens),
            'attention_mask': MagicMock(squeeze=lambda: tokens)
        }

    def __len__(self):
        return 1000

class CodeDatasetOriginal(Dataset):
    """Dataset for code training examples"""

    def __init__(self, data_file: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Only load from file if data_file is provided and not empty
        if data_file and data_file.strip():
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Dataset file not found: {data_file}")
            with open(data_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        else:
            # Empty dataset
            pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get('prompt', '')
        language = item.get('language', 'python')

        # Format prompt with language context
        formatted_prompt = f"Write high-quality {language} code:\n\n{prompt}\n\nCode:"

        # Tokenize
        encoding = self.tokenizer(
            formatted_prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'prompt': prompt,
            'language': language,
            'prompt_text': formatted_prompt
        }

class CodeDatasetOptimized(Dataset):
    """Dataset for code training examples"""

    def __init__(self, data_file: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Only load from file if data_file is provided and not empty
        if data_file and data_file.strip():
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Dataset file not found: {data_file}")
            with open(data_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        else:
            # Empty dataset
            pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get('prompt', '')
        language = item.get('language', 'python')

        # Format prompt with language context
        formatted_prompt = f"Write high-quality {language} code:\n\n{prompt}\n\nCode:"

        # Optimization: Skip tokenization as it's redundant (done later in generation/training)
        # We only need the prompt and language metadata

        return {
            'prompt': prompt,
            'language': language,
            'prompt_text': formatted_prompt
        }

def create_dummy_data(filename, num_samples=1000):
    with open(filename, 'w') as f:
        for i in range(num_samples):
            f.write(json.dumps({'prompt': f'Prompt {i}', 'language': 'python'}) + '\n')

def benchmark():
    data_file = 'dummy_train.jsonl'
    create_dummy_data(data_file)

    tokenizer = MockTokenizer()

    print("Benchmarking Original...")
    ds_orig = CodeDatasetOriginal(data_file, tokenizer)
    start = time.time()
    for i in range(len(ds_orig)):
        _ = ds_orig[i]
    end = time.time()
    print(f"Original Time: {end - start:.4f}s")

    print("Benchmarking Optimized...")
    ds_opt = CodeDatasetOptimized(data_file, tokenizer)
    start = time.time()
    for i in range(len(ds_opt)):
        _ = ds_opt[i]
    end = time.time()
    print(f"Optimized Time: {end - start:.4f}s")

    os.remove(data_file)

if __name__ == "__main__":
    benchmark()
