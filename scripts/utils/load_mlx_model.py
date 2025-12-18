#!/usr/bin/env python3
"""
Utility script to load and test MLX models for inference

This script demonstrates how to use the MLX-format models saved during training
for faster inference on Apple Silicon.
"""

import argparse
import sys
from pathlib import Path


def load_mlx_model(model_path: str, quantized: bool = False):
    """Load an MLX model for inference"""
    try:
        from mlx_lm import load, generate
        
        print(f"Loading MLX model from {model_path}...")
        model, tokenizer = load(model_path)
        print("Model loaded successfully!")
        
        return model, tokenizer
    
    except ImportError:
        print("ERROR: MLX libraries not installed.")
        print("Install with: pip install mlx mlx-lm")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)


def generate_code(model, tokenizer, prompt: str, language: str = "python", max_tokens: int = 512):
    """Generate code using MLX model"""
    try:
        from mlx_lm import generate
        
        full_prompt = f"Write high-quality {language} code:\n\n{prompt}\n\nCode:"
        
        print(f"\nGenerating {language} code...")
        print(f"Prompt: {prompt}\n")
        print("Generated code:")
        print("-" * 80)
        
        response = generate(
            model,
            tokenizer,
            prompt=full_prompt,
            max_tokens=max_tokens,
            temp=0.8,
            verbose=True
        )
        
        print(response)
        print("-" * 80)
        
        return response
    
    except Exception as e:
        print(f"ERROR during generation: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Load and test MLX model for inference")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to MLX model directory (e.g., ./checkpoints/checkpoint-500/mlx_model)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="Implement a binary search function",
        help='Code generation prompt'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='python',
        choices=['python', 'cpp', 'rust'],
        help='Programming language'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate'
    )
    
    args = parser.parse_args()
    
    # Check if model path exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}")
        print("\nTip: MLX models are saved in: ./checkpoints/checkpoint-{step}/mlx_model/")
        sys.exit(1)
    
    # Load model
    model, tokenizer = load_mlx_model(str(model_path))
    
    # Generate code
    generate_code(model, tokenizer, args.prompt, args.language, args.max_tokens)
    
    print("\n✓ MLX model inference completed successfully!")


if __name__ == "__main__":
    main()

