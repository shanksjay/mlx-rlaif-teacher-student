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
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_model_to_mlx(hf_path: str, mlx_path: str, quantize: str = None):
    """Convert HuggingFace model to MLX format"""
    # Preferred path (per Qwen docs): use the official CLI `mlx_lm.convert`.
    # This is the most compatible option across mlx-lm versions and supports q-bits flags.
    # Ref: https://qwen.readthedocs.io/en/latest/run_locally/mlx-lm.html
    def _run_cli():
        cmd = ["mlx_lm.convert", "--hf-path", hf_path, "--mlx-path", str(mlx_path)]
        if quantize:
            cmd.append("-q")
            if quantize == "q4_bit":
                cmd += ["--q-bits", "4"]
            elif quantize == "q8_bit":
                cmd += ["--q-bits", "8"]
        logger.info("Running: " + " ".join(cmd))
        subprocess.run(cmd, check=True)

    def _run_python_api():
        from mlx_lm import convert  # type: ignore
        convert(hf_path, str(mlx_path))
    
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
        logger.info("Converting model to MLX format (preferred: mlx_lm.convert)...")
        logger.info("This may take a few minutes...")

        try:
            _run_cli()
        except FileNotFoundError:
            logger.warning("mlx_lm.convert CLI not found on PATH; falling back to Python API.")
            _run_python_api()
        except subprocess.CalledProcessError as e:
            logger.warning(f"mlx_lm.convert failed (exit={e.returncode}); falling back to Python API.")
            _run_python_api()

        logger.info("✓ Model converted to MLX format")
        
        logger.info(f"\n✓ MLX model saved to: {mlx_path}")
        logger.info("\nUsage:")
        logger.info(f"  from mlx_lm import load, generate")
        logger.info(f"  model, tokenizer = load('{mlx_path}')")
        logger.info(f"  response = generate(model, tokenizer, prompt='...', max_tokens=512)")
        logger.info("\nOr update config.yaml:")
        logger.info("  hardware:")
        logger.info("    use_mlx_for_generation: true")
        logger.info(f"    mlx_model_path: {mlx_path}")
        if quantize:
            bits = 4 if quantize == "q4_bit" else 8
            logger.info("\nNote:")
            logger.info(f"  This MLX model was converted with quantization enabled (q-bits={bits}).")
        
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
  python convert_to_mlx.py --hf-path Qwen/Qwen2.5-7B-Instruct --mlx-path ./mlx_model/q4 --quantize q4_bit
  
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

