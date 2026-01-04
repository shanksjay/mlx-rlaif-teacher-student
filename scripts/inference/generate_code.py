#!/usr/bin/env python3
"""
Generate code from a trained model (checkpoint or MLX format)

This script can load either:
1. PyTorch checkpoints (with LoRA adapters) - uses the base model + adapter weights
2. MLX converted models - uses MLX for faster inference on Apple Silicon

Usage:
    # With MLX model
    python scripts/inference/generate_code.py \
        --model_path ./checkpoints/checkpoint-1000/mlx_model \
        --prompt "Implement a binary search function" \
        --language python

    # With PyTorch checkpoint
    python scripts/inference/generate_code.py \
        --model_path ./checkpoints/checkpoint-1000 \
        --prompt "Implement a binary search function" \
        --language python \
        --base_model Qwen/Qwen2.5-Coder-3B-Instruct
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

try:
    from mlx_lm import load as mlx_load, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def detect_model_type(model_path: str) -> str:
    """Detect if model path is MLX or PyTorch checkpoint"""
    path = Path(model_path)
    
    # Strong heuristic: if path contains "mlx_model", it's almost certainly MLX
    if "mlx_model" in str(model_path).lower() or "mlx" in str(path.name).lower():
        if MLX_AVAILABLE:
            return "mlx"
    
    # Check for MLX model indicators (mlx_lm.convert creates these files)
    # MLX models have: config.json, tokenizer files, and either:
    # - weights/ directory (older format)
    # - model.safetensors or weights.safetensors (newer format)
    has_mlx_config = (path / "config.json").exists()
    has_tokenizer = (
        (path / "tokenizer_config.json").exists() or
        (path / "tokenizer.json").exists()
    )
    has_mlx_weights = (
        (path / "weights").exists() or
        (path / "model.safetensors").exists() or
        (path / "weights.safetensors").exists() or
        (path / "model.npz").exists() or
        (path / "weights.npz").exists()
    )
    
    # Strong MLX indicators
    if has_mlx_config and has_tokenizer and has_mlx_weights:
        # Additional check: MLX models don't have PyTorch-specific files
        has_pytorch_files = (
            (path / "pytorch_model.bin").exists() or
            (path / "adapter_model.safetensors").exists() or
            (path / "adapter_model.bin").exists()
        )
        if not has_pytorch_files:
            return "mlx"
    
    # Check for PyTorch checkpoint indicators
    if (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists():
        return "pytorch_lora"
    
    if (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists():
        # But check if it's actually MLX (some MLX models have model.safetensors)
        if not has_mlx_config:
            return "pytorch_full"
    
    # If we have MLX indicators but also PyTorch files, prefer MLX if MLX_AVAILABLE
    if has_mlx_config and has_tokenizer and MLX_AVAILABLE:
        return "mlx"
    
    # Default: try MLX first if available, then PyTorch
    return "auto"


def load_mlx_model(model_path: str):
    """Load MLX model and tokenizer"""
    if not MLX_AVAILABLE:
        raise RuntimeError(
            "MLX libraries not available. Install with: pip install mlx mlx-lm"
        )
    
    print(f"Loading MLX model from {model_path}...")
    # Pass tokenizer_config to fix Mistral regex warning when applicable
    # This suppresses the warning about incorrect regex pattern in tokenizer
    tokenizer_config = {"fix_mistral_regex": True}
    model, tokenizer = mlx_load(model_path, tokenizer_config=tokenizer_config)
    print("✓ MLX model loaded successfully!")
    return model, tokenizer


def load_pytorch_model(
    model_path: str,
    base_model: Optional[str] = None,
    use_4bit: bool = True,
    device: str = "auto"
):
    """Load PyTorch checkpoint (with LoRA adapter if present)"""
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "Transformers library not available. Install with: pip install transformers peft"
        )
    
    path = Path(model_path)
    
    # Check if it's a LoRA checkpoint
    is_lora = (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
    
    if is_lora:
        if base_model is None:
            # Try to read from adapter_config.json
            try:
                import json
                with open(path / "adapter_config.json", "r") as f:
                    adapter_config = json.load(f)
                    base_model = adapter_config.get("base_model_name_or_path")
            except Exception:
                pass
            
            if base_model is None:
                raise ValueError(
                    "LoRA checkpoint detected but --base_model not provided. "
                    "Please specify the base model with --base_model."
                )
        
        print(f"Loading base model: {base_model}...")
        
        # Load base model
        if use_4bit and torch and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=quantization_config,
                device_map=device,
                trust_remote_code=True
            )
        else:
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch else None
            )
        
        print(f"Loading LoRA adapter from {model_path}...")
        model = PeftModel.from_pretrained(base_model_obj, str(path))
        print("✓ LoRA adapter loaded successfully!")
    else:
        # Full model checkpoint
        print(f"Loading full model from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch else None
        )
        print("✓ Full model loaded successfully!")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model if is_lora else str(path),
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def format_prompt(prompt: str, language: str = "python", use_thinking: bool = False) -> str:
    """Format prompt for code generation (matches training script format)
    
    Uses the same prompt format as the training script for consistency.
    
    Args:
        prompt: The code generation prompt
        language: Programming language
        use_thinking: If True, add thinking/reasoning step before code generation
    """
    if use_thinking:
        # Add thinking/reasoning step to improve output quality
        # Note: Don't start the code block in the prompt - let the model generate it
        formatted_prompt = f"""Write high-quality {language} code. Think through the problem step by step before writing the code.

Problem: {prompt}

Let's think through this step by step:
1. What are the requirements?
2. What edge cases should we consider?
3. What's the best approach?
4. How can we implement it efficiently?

Now write the code:"""
        return formatted_prompt
    else:
        system_prompt = f"You are an expert {language} programmer. Generate clean, efficient, and well-documented code."
        full_prompt = f"{prompt}\n\nGenerate high-quality {language} code:"
        return f"{system_prompt}\n\n{full_prompt}"


def extract_first_code_block(text: str) -> str:
    """Extract the first complete code block from generated text.
    
    This prevents repetitive output by stopping at the first complete code block.
    Handles patterns like:
    - ```language\n...code...\n```
    - Detects repetition patterns like "``` ```cpp" or duplicate code blocks
    
    Args:
        text: Generated text that may contain multiple code blocks
        
    Returns:
        First complete code block, or the text up to the first repetition
    """
    if not text:
        return text
    
    import re
    
    # Pattern to match code blocks: ```language\n...code...\n```
    # Also handles cases where there might be whitespace
    code_block_pattern = r'```\s*(\w+)?\s*\n(.*?)\n\s*```'
    matches = list(re.finditer(code_block_pattern, text, re.DOTALL))
    
    if matches:
        # Get the first complete code block
        first_match = matches[0]
        end_pos = first_match.end()
        
        # Extract the first block with its markdown
        first_block = text[:end_pos].strip()
        
        # Check if there's immediate repetition (like "``` ```cpp" right after)
        remaining = text[end_pos:].strip()
        
        # Pattern to detect repetition: closing ``` followed immediately by opening ```
        repetition_pattern = r'```\s*```'
        if re.search(repetition_pattern, remaining[:50]):  # Check first 50 chars
            # This is repetition, return only first block
            return first_block
        
        # If there are multiple blocks and the second starts very close, it's likely repetition
        if len(matches) > 1:
            second_match = matches[1]
            gap = second_match.start() - end_pos
            # If gap is very small (< 20 chars), it's likely repetition
            if gap < 20:
                return first_block
        
        # Otherwise, return the first block (user can see it's complete)
        return first_block
    
    # If no markdown blocks found, try to find natural stopping points
    # Look for patterns that indicate the code is complete and repetition is starting
    
    lines = text.split('\n')
    result_lines = []
    seen_signatures = set()
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Detect repetition: if we see "``` ```" or similar patterns
        if line_stripped.startswith('```') and i > 10:
            # Check if we've already seen a complete code block
            # Look ahead a few lines to see if this is starting a duplicate
            if i + 3 < len(lines):
                next_lines = '\n'.join(lines[i+1:i+4])
                # If next lines contain class/function definitions we've seen, it's repetition
                for sig in seen_signatures:
                    if sig in next_lines and len(result_lines) > 30:
                        # Repetition detected, stop here
                        return '\n'.join(result_lines).strip()
        
        # Track function/class signatures to detect repetition
        if line_stripped:
            # Extract signature patterns
            if 'template' in line_stripped or 'class ' in line_stripped or 'struct ' in line_stripped:
                # Extract the class/struct name
                match = re.search(r'(?:class|struct|template)\s+(\w+)', line_stripped)
                if match:
                    sig = match.group(1)
                    if sig in seen_signatures and len(result_lines) > 50:
                        # We've seen this class before, likely repetition
                        return '\n'.join(result_lines).strip()
                    seen_signatures.add(sig)
            elif line_stripped.startswith(('int main', 'void ', 'bool ', 'T ')):
                # Function signatures
                sig = line_stripped.split('(')[0] if '(' in line_stripped else line_stripped
                if sig in seen_signatures and len(result_lines) > 50:
                    return '\n'.join(result_lines).strip()
                seen_signatures.add(sig)
        
        result_lines.append(line)
        
        # Safety: if we've generated a lot and see clear repetition markers, stop
        if i > 200 and '```' in line_stripped:
            # Check if this looks like starting a new duplicate block
            if i + 5 < len(lines):
                upcoming = '\n'.join(lines[i:i+6])
                # Count how many class/function definitions we see
                def_count = len(re.findall(r'(?:class|struct|template|int main)', upcoming))
                if def_count > 2:  # Multiple definitions = likely repetition
                    return '\n'.join(result_lines[:i]).strip()
    
    return '\n'.join(result_lines).strip()


def extract_code_from_thinking(text: str, language: str) -> str:
    """Extract code from thinking output that may include reasoning before code.
    
    Looks for code blocks (```language ... ```) or code after thinking/reasoning text.
    """
    if not text:
        return text
    
    import re
    
    # Try to find code block first (with closing ```)
    code_block_pattern = rf"```{re.escape(language)}?\s*\n(.*?)```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try generic code block (with closing ```)
    code_block_pattern = r"```\s*\n(.*?)```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try to find code block that starts with ```{language} but doesn't close
    # (model generated code directly after opening fence)
    code_block_start_pattern = rf"```{re.escape(language)}?\s*\n(.*)"
    match = re.search(code_block_start_pattern, text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Remove any trailing ``` if present
        code = re.sub(r'```\s*$', '', code, flags=re.MULTILINE)
        if code:
            return code
    
    # Try generic code block start (without closing)
    code_block_start_pattern = r"```\s*\n(.*)"
    match = re.search(code_block_start_pattern, text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Remove any trailing ``` if present
        code = re.sub(r'```\s*$', '', code, flags=re.MULTILINE)
        if code:
            return code
    
    # If no code block, look for code-like patterns after thinking markers
    # Common patterns: "Now write the code:", "Implementation:", "Code:", etc.
    thinking_markers = [
        r"Now write the code:?\s*\n",
        r"Implementation:?\s*\n",
        r"Code:?\s*\n",
        r"Solution:?\s*\n",
        r"Here's the code:?\s*\n",
    ]
    
    for marker in thinking_markers:
        parts = re.split(marker, text, flags=re.IGNORECASE)
        if len(parts) > 1:
            # Take everything after the marker
            code = "\n".join(parts[1:]).strip()
            # Remove any code fences if present
            code = re.sub(r'^```\w*\s*\n', '', code, flags=re.MULTILINE)
            code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
            if code:
                return code
    
    # If no clear separation, return the text as-is (might be pure code)
    return text.strip()


def generate_with_mlx(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    language: str = "python",
    verbose: bool = True,
    use_thinking: bool = False
) -> Tuple[str, float]:
    """Generate code using MLX model"""
    formatted_prompt = format_prompt(prompt, language, use_thinking=use_thinking)
    
    # Create sampler if temperature > 0
    # Add repetition penalty to reduce repetitive output
    sampler = None
    if temperature > 0:
        try:
            # Try to add repetition_penalty if supported (helps prevent repetition)
            # MLX's make_sampler may support repetition_penalty parameter
            try:
                sampler = make_sampler(
                    temp=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=1.15  # Penalize repetition (1.0 = no penalty, >1.0 = penalize)
                )
            except TypeError:
                # If repetition_penalty not supported, use basic sampler
                sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        except Exception:
            sampler = None
    
    start_time = time.time()
    
    if verbose:
        print(f"\nGenerating code (max_tokens={max_tokens}, temp={temperature})...")
        print(f"Prompt: {prompt}\n")
        print("Generated code:")
        print("-" * 80)
    
    # Generate with verbose=False to avoid duplicate output
    # mlx_generate returns the full text (prompt + generated), so we need to extract just the generated part
    # Note: MLX generate doesn't support stop sequences directly, so we'll post-process to remove repetition
    generated_text = mlx_generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False  # Set to False to avoid duplicate output
    )
    
    generation_time = time.time() - start_time
    
    # Extract only the generated portion (remove the prompt)
    if formatted_prompt in generated_text:
        # Find where the prompt ends and extract only the generated part
        generated_only = generated_text.split(formatted_prompt, 1)[1].strip()
    else:
        # If prompt not found, assume the entire text is generated
        generated_only = generated_text.strip()
    
    # Extract code from thinking output if thinking was used
    if use_thinking:
        original_generated = generated_only
        extracted = extract_code_from_thinking(generated_only, language)
        # Only use extracted if it's not empty and different from original
        if extracted and extracted.strip() and extracted != original_generated:
            generated_only = extracted
        # If extraction didn't find anything but we have content, keep original
        # (might be pure code without markdown fences, or thinking text that we want to show)
        if not generated_only.strip() and original_generated.strip():
            # Fallback: return original if extraction removed everything
            generated_only = original_generated
    else:
        # Extract only the first complete code block to prevent repetition
        if generated_only.strip():
            generated_only = extract_first_code_block(generated_only)
    
    if verbose:
        print(generated_only)
        print("-" * 80)
    
    return generated_only, generation_time


def generate_with_pytorch(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    language: str = "python",
    device: str = "auto",
    use_thinking: bool = False
) -> Tuple[str, float]:
    """Generate code using PyTorch model"""
    formatted_prompt = format_prompt(prompt, language, use_thinking=use_thinking)
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move to device
    if device == "auto":
        if torch and torch.cuda.is_available():
            device = "cuda"
        elif torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    model.eval()
    
    print(f"\nGenerating code (max_tokens={max_tokens}, temp={temperature})...")
    print(f"Prompt: {prompt}\n")
    print("Generated code:")
    print("-" * 80)
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            top_k=top_k if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,  # Penalize repetition (1.0 = no penalty, >1.0 = penalize)
        )
    
    generation_time = time.time() - start_time
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the output
    if formatted_prompt in generated_text:
        generated_text = generated_text.split(formatted_prompt, 1)[1].strip()
    
    # Extract code from thinking output if thinking was used
    if use_thinking:
        original_generated = generated_text
        extracted = extract_code_from_thinking(generated_text, language)
        # Only use extracted if it's not empty and different from original
        if extracted and extracted.strip() and extracted != original_generated:
            generated_text = extracted
        # If extraction didn't find anything but we have content, keep original
        if not generated_text.strip() and original_generated.strip():
            # Fallback: return original if extraction removed everything
            generated_text = original_generated
    
    print(generated_text)
    print("-" * 80)
    
    return generated_text, generation_time


def main():
    parser = argparse.ArgumentParser(
        description="Generate code from a trained model (checkpoint or MLX format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With MLX model
  python scripts/inference/generate_code.py \\
      --model_path ./checkpoints/checkpoint-1000/mlx_model \\
      --prompt "Implement a binary search function"

  # With PyTorch checkpoint (LoRA)
  python scripts/inference/generate_code.py \\
      --model_path ./checkpoints/checkpoint-1000 \\
      --base_model Qwen/Qwen2.5-Coder-3B-Instruct \\
      --prompt "Implement a binary search function"

  # Interactive mode
  python scripts/inference/generate_code.py \\
      --model_path ./checkpoints/checkpoint-1000/mlx_model \\
      --interactive

  # Compare baseline vs checkpoint model
  python scripts/inference/generate_code.py \\
      --model_path ./checkpoints/checkpoint-1000/mlx_model \\
      --baseline_model_path ./mlx_model/q4 \\
      --prompt "Implement a binary search function" \\
      --language python

  # Compare with PyTorch models
  python scripts/inference/generate_code.py \\
      --model_path ./checkpoints/checkpoint-1000 \\
      --base_model Qwen/Qwen2.5-Coder-3B-Instruct \\
      --baseline_model_path Qwen/Qwen2.5-Coder-3B-Instruct \\
      --prompt "Implement a binary search function"
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model (MLX model directory or PyTorch checkpoint directory)"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name for LoRA checkpoints (e.g., Qwen/Qwen2.5-Coder-3B-Instruct). "
             "Auto-detected from adapter_config.json if not provided."
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Code generation prompt"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        choices=["python", "cpp", "rust", "javascript", "java"],
        help="Programming language"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 = greedy, >0.0 = sampling)"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization for PyTorch models (CUDA only)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for PyTorch models"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (prompt for input)"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "mlx", "pytorch"],
        help="Force model type (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--baseline_model_path",
        type=str,
        default=None,
        help="Path to baseline model (for comparison). If provided, generates with baseline first, then checkpoint."
    )
    
    parser.add_argument(
        "--baseline_base_model",
        type=str,
        default=None,
        help="Base model name for baseline LoRA checkpoint (if baseline is LoRA)"
    )
    
    parser.add_argument(
        "--use_thinking",
        action="store_true",
        default=False,
        help="Enable thinking/reasoning step before code generation to improve output quality. "
             "The model will think through the problem before writing code."
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}")
        sys.exit(1)
    
    # Validate baseline model path if provided
    # Baseline can be a local path or a HuggingFace model ID
    baseline_model_path = None
    baseline_is_hf_id = False
    if args.baseline_model_path:
        baseline_path_str = args.baseline_model_path
        # Check if it's a HuggingFace model ID (contains '/' but not a local path)
        if '/' in baseline_path_str and not Path(baseline_path_str).exists():
            # Likely a HuggingFace model ID
            baseline_is_hf_id = True
            baseline_model_path = baseline_path_str
            print(f"Baseline model: HuggingFace model ID '{baseline_path_str}' (will be loaded from HuggingFace)")
        else:
            # Local path
            baseline_model_path = Path(baseline_path_str)
            if not baseline_model_path.exists():
                print(f"ERROR: Baseline model path does not exist: {baseline_model_path}")
                sys.exit(1)
    
    # Detect model type
    if args.model_type == "auto":
        model_type = detect_model_type(str(model_path))
    else:
        model_type = args.model_type
    
    # Detect baseline model type if provided
    baseline_model_type = None
    if baseline_model_path:
        if args.model_type == "auto":
            baseline_model_type = detect_model_type(str(baseline_model_path))
        else:
            baseline_model_type = args.model_type
    
    # Load model
    try:
        use_mlx = False
        
        # Try MLX first if:
        # 1. Explicitly requested (model_type == "mlx")
        # 2. Auto-detected as MLX
        # 3. Auto mode and MLX is available (try MLX first, fallback to PyTorch)
        if model_type == "mlx" or (model_type == "auto" and MLX_AVAILABLE):
            try:
                print(f"Attempting to load as MLX model...")
                model, tokenizer = load_mlx_model(str(model_path))
                use_mlx = True
                print("✓ Successfully loaded as MLX model")
            except Exception as e:
                if model_type == "mlx":
                    # Explicitly requested MLX, so fail
                    raise
                # Auto mode: try PyTorch as fallback
                print(f"Warning: Failed to load as MLX model: {e}")
                if model_type == "auto":
                    print("Trying PyTorch format...")
                    use_mlx = False
                else:
                    raise
        
        # Load as PyTorch if MLX failed or wasn't attempted
        if not use_mlx:
            if model_type == "mlx":
                raise RuntimeError("MLX model loading failed and no fallback available")
            
            # Before loading PyTorch, do a final check: if this looks like an MLX model
            # but detection failed, try MLX one more time
            if model_type == "auto" and MLX_AVAILABLE:
                path = Path(model_path)
                # Check if it has MLX-like structure but detection missed it
                if (path / "config.json").exists():
                    try:
                        print("Re-attempting MLX load (detection may have missed MLX format)...")
                        model, tokenizer = load_mlx_model(str(model_path))
                        use_mlx = True
                        print("✓ Successfully loaded as MLX model (on retry)")
                    except Exception:
                        # MLX failed, proceed with PyTorch
                        pass
            
            if not use_mlx:
                print(f"Loading as PyTorch model...")
                model, tokenizer = load_pytorch_model(
                    str(model_path),
                    base_model=args.base_model,
                    use_4bit=args.use_4bit,
                    device=args.device
                )
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 80)
        print("Interactive Code Generation Mode")
        print("=" * 80)
        print("Enter prompts (or 'quit' to exit):\n")
        
        while True:
            try:
                prompt = input("Prompt: ").strip()
                if not prompt or prompt.lower() in ["quit", "exit", "q"]:
                    break
                
                if use_mlx:
                    generated, gen_time = generate_with_mlx(
                        model, tokenizer, prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        language=args.language,
                        verbose=True,
                        use_thinking=args.use_thinking
                    )
                else:
                    generated, gen_time = generate_with_pytorch(
                        model, tokenizer, prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        language=args.language,
                        device=args.device,
                        use_thinking=args.use_thinking
                    )
                
                # Calculate tokens per second
                num_tokens = len(tokenizer.encode(generated))
                tokens_per_sec = num_tokens / gen_time if gen_time > 0 else 0
                
                print(f"\nGeneration stats: {gen_time:.2f}s, {num_tokens} tokens, {tokens_per_sec:.1f} tokens/s\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n✓ Interactive mode ended.")
        return
    
    # Single prompt mode
    if args.prompt is None:
        print("ERROR: --prompt is required (or use --interactive mode)")
        sys.exit(1)
    
    # Load baseline model if provided
    baseline_model = None
    baseline_tokenizer = None
    baseline_use_mlx = False
    
    if baseline_model_path:
        print("\n" + "=" * 80)
        print("Loading Baseline Model for Comparison")
        print("=" * 80)
        try:
            if baseline_is_hf_id:
                # Baseline is a HuggingFace model ID - load as PyTorch model
                print(f"Loading baseline from HuggingFace: {baseline_model_path}")
                baseline_model, baseline_tokenizer = load_pytorch_model(
                    baseline_model_path,  # HuggingFace model ID
                    base_model=None,  # Not a LoRA checkpoint
                    use_4bit=args.use_4bit,
                    device=args.device
                )
                baseline_use_mlx = False
                print("✓ Baseline PyTorch model loaded from HuggingFace")
            else:
                # Baseline is a local path - try MLX first, then PyTorch
                if baseline_model_type == "mlx" or (baseline_model_type == "auto" and MLX_AVAILABLE):
                    try:
                        print(f"Loading baseline as MLX model...")
                        baseline_model, baseline_tokenizer = load_mlx_model(str(baseline_model_path))
                        baseline_use_mlx = True
                        print("✓ Baseline MLX model loaded")
                    except Exception as e:
                        if baseline_model_type == "mlx":
                            raise
                        print(f"Warning: Failed to load baseline as MLX: {e}")
                        baseline_use_mlx = False
                
                if not baseline_use_mlx:
                    print(f"Loading baseline as PyTorch model...")
                    baseline_model, baseline_tokenizer = load_pytorch_model(
                        str(baseline_model_path),
                        base_model=args.baseline_base_model or args.base_model,
                        use_4bit=args.use_4bit,
                        device=args.device
                    )
                    print("✓ Baseline PyTorch model loaded")
        except Exception as e:
            print(f"ERROR: Failed to load baseline model: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with checkpoint model only...")
            baseline_model = None
    
    try:
        # Generate with checkpoint/optimized model first
        print("\n" + "=" * 80)
        print("CHECKPOINT/OPTIMIZED MODEL OUTPUT")
        print("=" * 80)
        print(f"Using checkpoint model from: {model_path}")
        
        if use_mlx:
            generated, gen_time = generate_with_mlx(
                model, tokenizer, args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                language=args.language,
                verbose=True,
                use_thinking=args.use_thinking
            )
        else:
            generated, gen_time = generate_with_pytorch(
                model, tokenizer, args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                language=args.language,
                device=args.device,
                use_thinking=args.use_thinking
            )
        
        # Calculate tokens per second
        num_tokens = len(tokenizer.encode(generated))
        tokens_per_sec = num_tokens / gen_time if gen_time > 0 else 0
        
        print(f"\nCheckpoint Generation stats:")
        print(f"  Time: {gen_time:.2f}s")
        print(f"  Tokens: {num_tokens}")
        print(f"  Speed: {tokens_per_sec:.1f} tokens/s")
        
        # Generate with baseline if provided (for comparison)
        if baseline_model is not None:
            print("\n" + "=" * 80)
            print("BASELINE MODEL OUTPUT")
            print("=" * 80)
            print(f"Using baseline model from: {baseline_model_path}")
            
            if baseline_use_mlx:
                baseline_generated, baseline_gen_time = generate_with_mlx(
                    baseline_model, baseline_tokenizer, args.prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    language=args.language,
                    verbose=True,
                    use_thinking=args.use_thinking
                )
            else:
                baseline_generated, baseline_gen_time = generate_with_pytorch(
                    baseline_model, baseline_tokenizer, args.prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    language=args.language,
                    device=args.device,
                    use_thinking=args.use_thinking
                )
            
            baseline_num_tokens = len(baseline_tokenizer.encode(baseline_generated))
            baseline_tokens_per_sec = baseline_num_tokens / baseline_gen_time if baseline_gen_time > 0 else 0
            
            print(f"\nBaseline Generation stats:")
            print(f"  Time: {baseline_gen_time:.2f}s")
            print(f"  Tokens: {baseline_num_tokens}")
            print(f"  Speed: {baseline_tokens_per_sec:.1f} tokens/s")
        
        # Comparison summary if baseline was used
        if baseline_model is not None:
            print("\n" + "=" * 80)
            print("COMPARISON SUMMARY")
            print("=" * 80)
            print(f"Baseline:  {baseline_num_tokens} tokens, {baseline_gen_time:.2f}s, {baseline_tokens_per_sec:.1f} tok/s")
            print(f"Checkpoint: {num_tokens} tokens, {gen_time:.2f}s, {tokens_per_sec:.1f} tok/s")
            if baseline_gen_time > 0 and gen_time > 0:
                speedup = baseline_gen_time / gen_time
                if speedup > 1:
                    print(f"Speedup: {speedup:.2f}x (checkpoint faster than baseline)")
                elif speedup < 1:
                    slowdown = 1.0 / speedup
                    print(f"Slowdown: {slowdown:.2f}x (baseline faster than checkpoint)")
                else:
                    print("Speed: Same (checkpoint and baseline have similar speed)")
        
        print("\n✓ Code generation completed successfully!")
        
    except Exception as e:
        print(f"ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

