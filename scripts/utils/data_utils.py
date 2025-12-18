#!/usr/bin/env python3
"""
Data utilities for preparing and processing code training datasets
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def create_sample_dataset(output_file: str = "./data/train.jsonl", num_samples: int = 100):
    """
    Create a sample dataset for testing purposes.
    In production, you would use real code datasets.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Sample prompts for different languages
    sample_prompts = {
        "python": [
            "Implement a function to find the longest common subsequence between two strings",
            "Write a function to merge two sorted linked lists",
            "Create a decorator that measures function execution time",
            "Implement a binary search tree with insert, delete, and search operations",
            "Write a function to calculate the factorial of a number using memoization",
        ],
        "cpp": [
            "Implement a thread-safe singleton pattern in C++",
            "Write a template function to find the maximum element in a container",
            "Create a RAII wrapper for file handling",
            "Implement a custom smart pointer class",
            "Write a function to reverse a linked list iteratively",
        ],
        "rust": [
            "Implement a function to find all prime numbers up to n using Sieve of Eratosthenes",
            "Create a generic function to sort a vector in-place",
            "Write a function to parse a JSON string safely",
            "Implement a simple HTTP client using async/await",
            "Create a function to calculate the Fibonacci sequence using an iterator",
        ]
    }
    
    data = []
    for language, prompts in sample_prompts.items():
        for i, prompt in enumerate(prompts):
            for j in range(num_samples // (len(sample_prompts) * len(prompts)) + 1):
                if len(data) >= num_samples:
                    break
                data.append({
                    "prompt": prompt,
                    "language": language,
                    "id": f"{language}_{i}_{j}"
                })
            if len(data) >= num_samples:
                break
        if len(data) >= num_samples:
            break
    
    # Write to JSONL file
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Created sample dataset with {len(data)} examples at {output_file}")
    return output_file


def load_code_dataset(file_path: str) -> List[Dict]:
    """Load code dataset from JSONL file"""
    data = []
    if not os.path.exists(file_path):
        logger.warning(f"Dataset file not found: {file_path}")
        return data
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line: {e}")
    
    logger.info(f"Loaded {len(data)} examples from {file_path}")
    return data


def validate_dataset(data: List[Dict]) -> bool:
    """Validate dataset format"""
    required_fields = ['prompt', 'language']
    valid_languages = ['python', 'cpp', 'rust']
    
    for item in data:
        for field in required_fields:
            if field not in item:
                logger.error(f"Missing required field: {field}")
                return False
        
        if item['language'] not in valid_languages:
            logger.warning(f"Invalid language: {item['language']}")
    
    return True


if __name__ == "__main__":
    # Create sample datasets
    create_sample_dataset("./data/train.jsonl", num_samples=100)
    create_sample_dataset("./data/eval.jsonl", num_samples=20)

