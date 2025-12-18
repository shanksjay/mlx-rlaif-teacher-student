#!/usr/bin/env python3
"""
Convert PyTorch model to MLX format for faster inference on Apple Silicon

MLX provides 5-10x faster inference than PyTorch MPS on Apple Silicon by
leveraging both GPU and Neural Engine.
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_model_to_mlx(hf_path: str, mlx_path: str, quantize: str = None):
    """Convert HuggingFace model to MLX format"""
    try:
        from mlx_lm import convert
        # Try to import quantize from different locations
        mlx_quantize = None
        try:
            from mlx_lm import quantize as mlx_quantize
        except ImportError:
            try:
                from mlx_lm.utils import quantize as mlx_quantize
            except ImportError:
                try:
                    from mlx_lm.quantize import quantize as mlx_quantize
                except ImportError:
                    pass  # Quantization not available, will note in output
    except ImportError as e:
        logger.error(f"MLX libraries not installed. Error: {e}")
        logger.error("Install with:")
        logger.error("  uv pip install mlx mlx-lm")
        return False
    
    logger.info("="*80)
    logger.info("Converting Model to MLX Format")
    logger.info("="*80)
    logger.info(f"Source: {hf_path}")
    logger.info(f"Destination: {mlx_path}")
    if quantize:
        logger.info(f"Quantization: {quantize}")
    logger.info("="*80)
    
    # Handle existing directory
    mlx_path = Path(mlx_path)
    if mlx_path.exists():
        logger.warning(f"Output path already exists: {mlx_path}")
        import shutil
        try:
            # Try to delete automatically (non-interactive)
            shutil.rmtree(mlx_path)
            logger.info(f"Deleted existing directory: {mlx_path}")
        except Exception as e:
            logger.error(f"Could not delete existing directory: {e}")
            logger.info("Please delete it manually or specify a different path.")
            return False
    
    try:
        # Convert model
        logger.info("Converting model to MLX format...")
        logger.info("This may take a few minutes...")
        
        convert(hf_path, str(mlx_path))
        logger.info("✓ Model converted to MLX format")
        
        # Apply quantization if specified
        # Note: In MLX, quantization is typically done during loading, not conversion
        # But we'll try to apply it if the function is available
        if quantize:
            if mlx_quantize:
                logger.info(f"Applying {quantize} quantization...")
                if quantize == "q4_bit":
                    bits = 4
                elif quantize == "q8_bit":
                    bits = 8
                else:
                    logger.warning(f"Unknown quantization: {quantize}")
                    return True
                
                try:
                    # Try to quantize (may not work with all mlx_lm versions)
                    mlx_quantize(str(mlx_path), str(mlx_path), bits=bits)
                    logger.info(f"✓ Model quantized to {bits}-bit")
                except Exception as e:
                    logger.warning(f"Quantization during conversion failed: {e}")
                    logger.info("Model converted but not quantized.")
                    logger.info(f"Note: You can use quantization when loading the model:")
                    logger.info(f"  from mlx_lm import load")
                    logger.info(f"  model, tokenizer = load('{mlx_path}', quantize={bits})")
            else:
                logger.info(f"Quantization requested: {quantize}")
                logger.info("Note: Quantization in MLX is typically done during model loading, not conversion.")
                logger.info("The model has been converted. To use quantization:")
                logger.info(f"  from mlx_lm import load")
                bits = 4 if quantize == "q4_bit" else 8
                logger.info(f"  model, tokenizer = load('{mlx_path}', quantize={bits})")
        
        logger.info(f"\n✓ MLX model saved to: {mlx_path}")
        logger.info("\nUsage:")
        logger.info(f"  from mlx_lm import load, generate")
        logger.info(f"  model, tokenizer = load('{mlx_path}')")
        logger.info(f"  response = generate(model, tokenizer, prompt='...', max_tokens=512)")
        logger.info("\nOr update config.yaml:")
        logger.info("  hardware:")
        logger.info("    use_mlx_for_generation: true")
        logger.info(f"    mlx_model_path: {mlx_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error converting model: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to MLX format for faster inference",
        epilog="""
Examples:
  # Convert from HuggingFace
  python convert_to_mlx.py --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ./mlx_model
  
  # Convert with quantization
  python convert_to_mlx.py --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ./mlx_model --quantize q4_bit
  
  # Convert from local checkpoint
  python convert_to_mlx.py --hf-path ./checkpoints/checkpoint-500 --mlx-path ./mlx_model
        """
    )
    
    parser.add_argument(
        '--hf-path',
        type=str,
        required=True,
        help='HuggingFace model path (ID or local path)'
    )
    
    parser.add_argument(
        '--mlx-path',
        type=str,
        required=True,
        help='Output path for MLX model'
    )
    
    parser.add_argument(
        '--quantize',
        type=str,
        choices=['q4_bit', 'q8_bit'],
        default=None,
        help='Quantization level (q4_bit or q8_bit)'
    )
    
    args = parser.parse_args()
    
    success = convert_model_to_mlx(args.hf_path, args.mlx_path, args.quantize)
    
    if success:
        logger.info("\n✓ Conversion complete!")
        return 0
    else:
        logger.error("\n✗ Conversion failed!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

