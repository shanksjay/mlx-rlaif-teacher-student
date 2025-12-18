#!/usr/bin/env python3
"""
Preload model script to cache model weights locally for faster subsequent loads

This script downloads and caches the model locally, which speeds up training startup.
It supports both PyTorch (for training) and MLX (for faster inference) formats.

Usage:
    # Preload PyTorch model
    uv run python preload_model.py --model Qwen/Qwen2.5-7B-Instruct
    
    # Preload and test with MLX model (5-10x faster generation)
    # Note: MLX model must be converted first with quantization
    uv run python scripts/utils/convert_to_mlx.py --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ./mlx_model/q8 --quantize q8_bit
    uv run python scripts/utils/preload_model.py --model Qwen/Qwen2.5-7B-Instruct --mlx-path ./mlx_model/q8 --mlx-quantize q8_bit
    
Note: MLX quantization is applied during conversion, not loading. The --mlx-quantize flag
is informational and indicates what quantization level the model should have.
"""

import argparse
import logging
import time
import threading
from pathlib import Path

import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable tokenizer parallelism warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def preload_model(
    model_name: str,
    use_4bit: bool = True,
    use_mps: bool = True,
    cache_dir: str = None,
    mlx_model_path: str = None,
    mlx_quantization: str = None,
    skip_pytorch: bool = False
):
    """Preload and cache model for faster subsequent loads
    
    Args:
        model_name: HuggingFace model name
        use_4bit: Use 4-bit quantization for PyTorch model
        use_mps: Use MPS backend if available
        cache_dir: Custom cache directory
        mlx_model_path: Path to MLX model (if already converted)
        mlx_quantization: MLX quantization level (q4_bit or q8_bit)
        skip_pytorch: Skip PyTorch model loading if MLX is available
    """
    
    logger.info(f"Preloading model: {model_name}")
    logger.info("This will download the model if not already cached.")
    logger.info("Subsequent training runs will be faster.\n")
    
    start_time = time.time()
    
    # Check for MLX model first
    mlx_available = False
    skip_pytorch_if_mlx = False
    if mlx_model_path and os.path.exists(mlx_model_path):
        mlx_available = True
        skip_pytorch_if_mlx = True  # Skip PyTorch if MLX is explicitly provided
    
    # Load tokenizer (needed for both PyTorch and MLX)
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    logger.info("✓ Tokenizer loaded")
    
    # Load PyTorch model only if not skipping
    model = None
    load_time = 0
    shard_time = 0
    memory_samples = []
    process = None  # Initialize for later use
    
    if not skip_pytorch_if_mlx:
        # Load model with profiling
        logger.info("Loading PyTorch model weights...")
        logger.info("Profiling loading performance...")
        
        # Monitor memory (use both process and system memory)
        process = psutil.Process()
        system_mem = psutil.virtual_memory()
        process_mem_before = process.memory_info().rss / (1024 ** 3)
        system_mem_before = system_mem.used / (1024 ** 3)
        logger.info(f"Memory before loading:")
        logger.info(f"  Process RSS: {process_mem_before:.2f} GB")
        logger.info(f"  System used: {system_mem_before:.2f} GB / {system_mem.total / (1024 ** 3):.2f} GB")
        
        model_kwargs = {
            "device_map": "auto",
            "dtype": torch.bfloat16 if use_mps else torch.float32,  # Use dtype instead of torch_dtype
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        # Prefer safetensors for faster loading
        try:
            from safetensors.torch import load_file
            model_kwargs["use_safetensors"] = True
            logger.info("Using safetensors format for faster loading")
        except ImportError:
            logger.info("Using standard format (safetensors not available)")
        
        if use_4bit:
            logger.info("Using 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["dtype"] = torch.bfloat16  # Use dtype instead of torch_dtype
        
        # Monitor loading with background thread
        monitoring = True
        memory_samples = []
        shard_start = time.time()
        
        def monitor_loading():
            """Monitor memory during loading"""
            while monitoring:
                elapsed = time.time() - shard_start
                try:
                    process_mem = process.memory_info().rss / (1024 ** 3)
                    system_mem = psutil.virtual_memory().used / (1024 ** 3)
                    # Store both process and system memory
                    memory_samples.append((elapsed, process_mem, system_mem))
                except:
                    pass
                time.sleep(0.5)  # Sample every 0.5 seconds
        
        monitor_thread = threading.Thread(target=monitor_loading, daemon=True)
        monitor_thread.start()
        
        # Check if model is cached and verify cache usage
        # Get cache directory
        if cache_dir:
            hf_cache_dir = cache_dir
        else:
            # Get Hugging Face cache directory
            # Standard location: ~/.cache/huggingface or HF_HOME environment variable
            hf_cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        
        # Check if model files exist in cache
        model_cache_path = Path(hf_cache_dir) / "hub" / f"models--{model_name.replace('/', '--')}"
        is_cached = model_cache_path.exists() and (
            any(model_cache_path.rglob("*.safetensors")) or 
            any(model_cache_path.rglob("*.bin"))
        )
        
        if is_cached:
            logger.info(f"✓ Model cache found at: {model_cache_path}")
            logger.info("Loading from cache (should be faster than download)")
            # Check cache size
            try:
                cache_size = sum(f.stat().st_size for f in model_cache_path.rglob("*") if f.is_file()) / (1024 ** 3)
                logger.info(f"  Cache size: {cache_size:.2f} GB")
            except:
                pass
        else:
            logger.info("⚠️  Model not in cache. Will download first time.")
            logger.info(f"  Cache location: {model_cache_path}")
        
        # Clear cache before loading
        if use_mps and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
            cache_dir=cache_dir
        )
        
        monitoring = False
        monitor_thread.join(timeout=2)
        
        load_time = time.time() - start_time
        shard_time = time.time() - shard_start
    else:
        logger.info("Skipping PyTorch model loading (using MLX instead)")
        load_time = time.time() - start_time
        shard_time = 0
        memory_samples = []  # Initialize empty for later checks
    
    # Analyze loading performance (only if PyTorch model was loaded)
    if not skip_pytorch and memory_samples and len(memory_samples) > 4:
        logger.info("\n" + "="*60)
        logger.info("Loading Performance Analysis:")
        logger.info("="*60)
        
        quarter_size = len(memory_samples) // 4
        quarter_times = []
        
        for i in range(4):
            start_idx = i * quarter_size
            end_idx = (i + 1) * quarter_size if i < 3 else len(memory_samples)
            
            if start_idx < len(memory_samples):
                quarter_data = memory_samples[start_idx:end_idx]
                if quarter_data:
                    time_start = quarter_data[0][0]
                    time_end = quarter_data[-1][0]
                    # Handle both old format (2 values) and new format (3 values)
                    if len(quarter_data[0]) == 3:
                        process_mem_start = quarter_data[0][1]
                        process_mem_end = quarter_data[-1][1]
                        system_mem_start = quarter_data[0][2]
                        system_mem_end = quarter_data[-1][2]
                        process_mem_peak = max(m[1] for m in quarter_data)
                        system_mem_peak = max(m[2] for m in quarter_data)
                        process_mem_delta = process_mem_end - process_mem_start
                        system_mem_delta = system_mem_end - system_mem_start
                    else:
                        # Old format compatibility
                        process_mem_start = quarter_data[0][1]
                        process_mem_end = quarter_data[-1][1]
                        system_mem_start = system_mem_end = system_mem_peak = 0
                        process_mem_peak = max(m[1] for m in quarter_data)
                        process_mem_delta = process_mem_end - process_mem_start
                        system_mem_delta = 0
                    
                    quarter_time = time_end - time_start
                    quarter_times.append(quarter_time)
                    
                    logger.info(
                        f"\n{i*25}%-{(i+1)*25}% (Shard {i+1}): "
                        f"Time={quarter_time:.1f}s"
                    )
                    logger.info(
                        f"  Process Memory: {process_mem_start:.2f}GB→{process_mem_end:.2f}GB "
                        f"(peak: {process_mem_peak:.2f}GB, Δ{process_mem_delta:+.2f}GB)"
                    )
                    if system_mem_delta != 0:
                        logger.info(
                            f"  System Memory: {system_mem_start:.2f}GB→{system_mem_end:.2f}GB "
                            f"(peak: {system_mem_peak:.2f}GB, Δ{system_mem_delta:+.2f}GB)"
                        )
        
        # Compare quarters
        if len(quarter_times) == 4:
            avg_first_three = sum(quarter_times[:3]) / 3
            last_quarter = quarter_times[3]
            if last_quarter > avg_first_three * 1.2:
                ratio = last_quarter / avg_first_three
                logger.info(f"\n⚠️  Last shard (75-100%) is {ratio:.1f}x slower than average")
                logger.info("   This is normal due to:")
                logger.info("   - Memory pressure from previous shards")
                logger.info("   - Device mapping finalization")
                logger.info("   - Quantization setup completion")
    
    # Monitor memory after loading (only if PyTorch model was loaded)
    if not skip_pytorch_if_mlx and process is not None:
        process_mem_after = process.memory_info().rss / (1024 ** 3)
        system_mem_after = psutil.virtual_memory().used / (1024 ** 3)
        process_mem_used = process_mem_after - process_mem_before
        system_mem_used = system_mem_after - system_mem_before
        
        logger.info(f"\nMemory after loading:")
        logger.info(f"  Process RSS: {process_mem_after:.2f} GB (Δ{process_mem_used:+.2f} GB)")
        logger.info(f"  System used: {system_mem_after:.2f} GB / {psutil.virtual_memory().total / (1024 ** 3):.2f} GB (Δ{system_mem_used:+.2f} GB)")
        logger.info(f"Shard loading time: {shard_time:.1f}s ({shard_time/60:.1f} minutes)")
        
        # Clear cache after loading
        if use_mps and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"\n✓ PyTorch model preloaded successfully!")
        logger.info(f"Total time: {load_time:.1f} seconds ({load_time/60:.1f} minutes)")
        logger.info(f"\nModel is now cached. Training startup will be faster.")
        
        # Warm up model for faster first generation
        # First generation is often slow due to MPS initialization
        logger.info("\nWarming up PyTorch model (first generation is slower)...")
        warmup_prompt = "Test"
        warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        warmup_inputs = {k: v.to(device) for k, v in warmup_inputs.items()}
        
        with torch.no_grad():
            try:
                # Very short warmup generation
                _ = model.generate(
                    **warmup_inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                logger.info("✓ PyTorch model warmed up")
            except Exception as e:
                logger.debug(f"Warmup failed (non-critical): {e}")
        
        # Clear cache after warmup
        if use_mps and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    else:
        logger.info(f"\n✓ Skipped PyTorch model loading (using MLX instead)")
        logger.info(f"Total time: {load_time:.1f} seconds")
    
    # Test a quick generation to verify it works
    logger.info("\nTesting model with a quick generation...")
    test_prompt = "Write a function to add two numbers:"
    
    # Try MLX first (much faster on Apple Silicon)
    use_mlx = False
    mlx_model = None
    mlx_tokenizer = None
    
    # Determine MLX model path
    mlx_path_to_check = None
    if mlx_model_path:
        mlx_path_to_check = mlx_model_path
    else:
        # Check common locations (new consolidated structure)
        possible_paths = [
            "./mlx_model/q8",  # Q8 quantized (best balance)
            "./mlx_model/q4",  # Q4 quantized (smallest)
            "./mlx_model/base", # Unquantized base model
            "./mlx_model/base", # Unquantized base model
            f"./mlx_{model_name.replace('/', '_')}",
            os.path.expanduser(f"~/.cache/mlx/{model_name.replace('/', '_')}")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                mlx_path_to_check = path
                break
    
    # Try to load MLX model if available
    if mlx_path_to_check and os.path.exists(mlx_path_to_check):
        try:
            from mlx_lm import load, generate as mlx_generate
            logger.info(f"Loading MLX model from: {mlx_path_to_check}")
            
            # Determine quantization bits
            quantize_bits = None
            if mlx_quantization:
                if mlx_quantization == "q4_bit":
                    quantize_bits = 4
                elif mlx_quantization == "q8_bit":
                    quantize_bits = 8
            
            # Load MLX model with optional quantization
            logger.info("Loading MLX model...")
            if quantize_bits:
                logger.info(f"Attempting to load with {quantize_bits}-bit quantization...")
                try:
                    # Try loading with quantization parameter (if supported)
                    mlx_model, mlx_tokenizer = load(mlx_path_to_check, quantize=quantize_bits)
                    logger.info(f"✓ Loaded with {quantize_bits}-bit quantization")
                except (TypeError, ValueError) as e:
                    # If quantize parameter not supported, load normally
                    logger.info(f"Quantization parameter not supported: {e}")
                    logger.info("Loading model normally (quantization may be in model weights)")
                    mlx_model, mlx_tokenizer = load(mlx_path_to_check)
                    if quantize_bits:
                        logger.info(f"Note: Model should have been converted with {quantize_bits}-bit quantization")
            else:
                mlx_model, mlx_tokenizer = load(mlx_path_to_check)
            
            use_mlx = True
            logger.info("✓ MLX model loaded successfully")
            if quantize_bits:
                logger.info(f"  Using {quantize_bits}-bit quantization for faster inference")
        except ImportError:
            logger.warning("MLX libraries not available. Install with: uv pip install mlx mlx-lm")
        except Exception as e:
            logger.warning(f"Could not load MLX model: {e}")
            logger.info("Falling back to PyTorch MPS for generation")
    elif mlx_model_path:
        # User specified MLX path but it doesn't exist
        logger.warning(f"MLX model path specified but not found: {mlx_model_path}")
        logger.info("Tip: Convert model to MLX format:")
        logger.info(f"  uv run python convert_to_mlx.py --hf-path {model_name} --mlx-path {mlx_model_path}")
    else:
        # No MLX model specified, suggest converting
        logger.info("No MLX model found. Using PyTorch MPS (slower).")
        logger.info("Tip: Convert model to MLX for 5-10x faster generation:")
        logger.info(f"  uv run python convert_to_mlx.py --hf-path {model_name} --mlx-path ./mlx_model/q8 --quantize q8_bit")
    
    # Optimize model for inference if on MPS (only if PyTorch model is loaded)
    if not use_mlx and model is not None:
        if use_mps and torch.backends.mps.is_available():
            # Compile model for faster inference (PyTorch 2.0+)
            try:
                if hasattr(torch, 'compile'):
                    logger.info("Compiling PyTorch model for faster inference...")
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("✓ Model compiled")
            except Exception as e:
                logger.debug(f"Model compilation not available: {e}")
    
    gen_start = time.time()
    
    if use_mlx and mlx_model:
        # Use MLX for generation (much faster)
        from mlx_lm import generate as mlx_generate
        # Optimize generation parameters for speed
        # MLX generate uses sampler parameter, not direct temp/top_k/top_p
        # For fastest generation, use minimal parameters
        generated_text = mlx_generate(
            mlx_model,
            mlx_tokenizer,
            prompt=test_prompt,
            max_tokens=20,
            # Note: MLX uses sampler for temperature/top_k/top_p control
            # For fastest: no sampler (greedy), or use default sampler with low temp
        )
        gen_time = time.time() - gen_start
        tokens_generated = len(mlx_tokenizer.encode(generated_text)) - len(mlx_tokenizer.encode(test_prompt))
        generated = generated_text
    else:
        # Use PyTorch MPS (only if model is loaded)
        if model is not None:
            # Prepare inputs
            inputs = tokenizer(test_prompt, return_tensors="pt")
            # Move to device with optimized settings
            device = next(model.parameters()).device
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Synchronize MPS before generation
                if device.type == "mps":
                    torch.mps.synchronize()
                
                # Optimize generation for speed - only pass valid parameters
                generation_config = {
                    "max_new_tokens": 20,  # Reduced for faster test
                    "do_sample": False,  # Greedy decoding (fastest)
                    "num_beams": 1,  # Single beam (fastest)
                    "pad_token_id": tokenizer.eos_token_id,
                    "use_cache": True,  # Enable KV cache for faster generation
                    "output_scores": False,  # Don't compute scores (faster)
                    "return_dict_in_generate": False,  # Return simple tensor (faster)
                }
                
                outputs = model.generate(**inputs, **generation_config)
                
                # Synchronize MPS after generation
                if device.type == "mps":
                    torch.mps.synchronize()
            
            gen_time = time.time() - gen_start
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        else:
            logger.warning("No model available for generation test")
            gen_time = 0
            generated = ""
            tokens_generated = 0
    
    tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0
    
    logger.info(f"Generation time: {gen_time:.2f}s")
    logger.info(f"Tokens generated: {tokens_generated} ({tokens_per_sec:.1f} tokens/sec)")
    logger.info(f"Test generation: {generated[:150]}...")
    
    if use_mlx:
        logger.info(f"✓ MLX generation completed ({tokens_per_sec:.1f} tokens/sec)")
        if tokens_per_sec > 1.0:
            logger.info("  Excellent performance with MLX!")
        elif tokens_per_sec > 0.5:
            logger.info("  Good performance with MLX")
        else:
            logger.warning("  MLX generation slower than expected")
            if mlx_quantization:
                logger.info("  Consider using a different quantization level or no quantization")
    elif gen_time > 10:
        logger.warning(f"⚠️  Generation is slow ({gen_time:.1f}s).")
        logger.warning("   PyTorch MPS is slower than MLX on Apple Silicon.")
        logger.info("\n💡 To improve generation speed (5-10x faster):")
        logger.info("   1. Convert model to MLX format:")
        logger.info(f"      uv run python convert_to_mlx.py --hf-path {model_name} --mlx-path ./mlx_model/q8 --quantize q8_bit")
        logger.info("   2. Run preload with MLX:")
        logger.info("      uv run python preload_model.py --model Qwen/Qwen2.5-7B-Instruct --mlx-path ./mlx_model/q8 --mlx-quantize q8_bit")
        logger.info("   3. Update config.yaml:")
        logger.info("      hardware:")
        logger.info("        use_mlx_for_generation: true")
        logger.info("        mlx_model_path: ./mlx_model/q8")
        logger.info("        mlx_quantization: q8_bit")
        logger.info("\n   MLX leverages Apple's GPU and Neural Engine for much faster inference.")
    elif tokens_per_sec < 1.0:
        logger.warning(f"⚠️  Very slow generation ({tokens_per_sec:.2f} tokens/sec)")
        logger.info("   Consider using MLX for 5-10x speedup:")
        logger.info(f"     uv run python convert_to_mlx.py --hf-path {model_name} --mlx-path ./mlx_model/q8 --quantize q8_bit")
    
    logger.info("\n✓ Model is ready for training!")


def main():
    parser = argparse.ArgumentParser(description="Preload model for faster training startup")
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Model name to preload'
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
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='Custom cache directory'
    )
    parser.add_argument(
        '--mlx-path',
        type=str,
        default=None,
        dest='mlx_model_path',
        help='Path to MLX model (if already converted)'
    )
    parser.add_argument(
        '--mlx-quantize',
        type=str,
        choices=['q4_bit', 'q8_bit'],
        default=None,
        dest='mlx_quantization',
        help='MLX quantization level (q4_bit or q8_bit)'
    )
    
    args = parser.parse_args()
    
    use_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    preload_model(
        model_name=args.model,
        use_4bit=args.use_4bit,
        use_mps=use_mps,
        cache_dir=args.cache_dir,
        mlx_model_path=args.mlx_model_path,
        mlx_quantization=args.mlx_quantization
    )


if __name__ == "__main__":
    main()

