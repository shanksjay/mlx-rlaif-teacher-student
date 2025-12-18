#!/usr/bin/env python3
"""
Standalone script to profile model loading with detailed call stack information

This script is designed to be profiled with Instruments or py-spy.
It loads the model and provides detailed timing information.
"""

import os
import sys
import time
import argparse
import logging
import traceback

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_with_profiling(model_name: str, use_4bit: bool = True):
    """Load model with detailed function-level profiling markers"""
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    use_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    logger.info("="*80)
    logger.info("Model Loading Profiling")
    logger.info("="*80)
    logger.info(f"Model: {model_name}")
    logger.info(f"4-bit Quantization: {use_4bit}")
    logger.info(f"MPS Available: {use_mps}")
    logger.info("="*80)
    
    # Phase 1: Tokenizer
    logger.info("\n[Phase 1] Loading tokenizer...")
    phase_start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True
    )
    
    phase_time = time.time() - phase_start
    logger.info(f"✓ Tokenizer loaded in {phase_time:.2f}s")
    
    # Phase 2: Model Config
    logger.info("\n[Phase 2] Loading model config...")
    phase_start = time.time()
    
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    phase_time = time.time() - phase_start
    logger.info(f"✓ Config loaded in {phase_time:.2f}s")
    
    # Phase 3: Model Weights
    logger.info("\n[Phase 3] Loading model weights...")
    phase_start = time.time()
    
    model_kwargs = {
        "device_map": "auto",
        "dtype": torch.bfloat16 if use_mps else torch.float32,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    
    # Prefer safetensors
    try:
        from safetensors.torch import load_file
        model_kwargs["use_safetensors"] = True
        logger.info("Using safetensors format")
    except ImportError:
        logger.info("Using standard format")
    
    if use_4bit:
        logger.info("Setting up 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["dtype"] = torch.bfloat16
    
    # Clear cache before loading
    if use_mps and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Calling AutoModelForCausalLM.from_pretrained()...")
    logger.info("(This is where most time is spent - profile this function)")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    phase_time = time.time() - phase_start
    logger.info(f"✓ Model weights loaded in {phase_time:.2f}s ({phase_time/60:.1f} minutes)")
    
    # Phase 4: Post-processing
    logger.info("\n[Phase 4] Post-processing...")
    phase_start = time.time()
    
    # Access model properties to trigger any lazy initialization
    _ = model.device
    _ = model.dtype
    _ = model.config
    
    phase_time = time.time() - phase_start
    logger.info(f"✓ Post-processing complete in {phase_time:.2f}s")
    
    # Clear cache after loading
    if use_mps and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    total_time = time.time() - phase_start
    logger.info(f"\n✓ Model loading complete!")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Profile model loading with detailed call stack information",
        epilog="""
This script is designed to be profiled with:
  - Apple Instruments: instruments -t "Time Profiler" -D trace.trace python profile_model_loading.py
  - py-spy: py-spy record -o flamegraph.svg -- python profile_model_loading.py
  - cProfile: python -m cProfile -o profile.prof profile_model_loading.py
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Model to profile'
    )
    
    parser.add_argument(
        '--use_4bit',
        action='store_true',
        default=True,
        help='Use 4-bit quantization'
    )
    
    parser.add_argument(
        '--no_4bit',
        dest='use_4bit',
        action='store_false',
        help='Disable 4-bit quantization'
    )
    
    args = parser.parse_args()
    
    try:
        model, tokenizer = load_model_with_profiling(
            model_name=args.model,
            use_4bit=args.use_4bit
        )
        
        logger.info("\n✓ Profiling complete!")
        logger.info("Check the profiling output for detailed call stack information.")
        
    except Exception as e:
        logger.error(f"Error during profiling: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

