#!/usr/bin/env python3
"""
Validation script to compare pre-training vs post-training model quality

This script generates code samples from both the baseline and fine-tuned models
and compares their quality using the teacher model for scoring.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import sys
from pathlib import Path as _Path
import warnings
import subprocess
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure repo root is on sys.path so `import scripts.*` works when running this file directly.
# When executed as `python scripts/validation/validate_model.py`, Python adds only the script directory to sys.path.
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Suppress a noisy upstream warning sometimes emitted by tooling/deps when running from a script path.
warnings.filterwarnings(
    "ignore",
    message=r"The module name .* is not a valid Python identifier\\. Please rename the original module to avoid import issues\\.",
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _truncate(s: str, n: int) -> str:
    """Truncate string to n chars for terminal output (0 = no truncation)."""
    try:
        n = int(n)
    except Exception:
        n = 0
    if n <= 0:
        return s
    if len(s) <= n:
        return s
    return s[:n] + "..."


class ModelValidator:
    """Validate and compare baseline vs fine-tuned models"""
    
    def __init__(
        self,
        base_model: str,
        fine_tuned_path: str,
        teacher_provider: str = "openai",
        teacher_model: str | None = None,
        *,
        use_mlx: bool = True,
        baseline_mlx_path: str = "./mlx_model/q4",
        fine_tuned_mlx_path: str | None = None,
        print_chars: int = 500,
    ):
        self.base_model_name = base_model
        self.fine_tuned_path = fine_tuned_path
        self.teacher_provider = teacher_provider
        # Default teacher model depends on provider.
        if teacher_model:
            self.teacher_model = teacher_model
        else:
            self.teacher_model = (
                "claude-3-5-haiku-20241022" if teacher_provider == "anthropic" else "gpt-4-turbo-preview"
            )

        self.use_mlx = bool(use_mlx)
        self._print_chars = int(print_chars) if int(print_chars) >= 0 else 0
        self.baseline_mlx_path = str(baseline_mlx_path)
        # Default fine-tuned MLX path: <checkpoint>/mlx_model if present, else treat fine_tuned_path as an MLX dir.
        ft_default = str(Path(fine_tuned_path) / "mlx_model")
        self.fine_tuned_mlx_path = str(fine_tuned_mlx_path or (ft_default if Path(ft_default).exists() else fine_tuned_path))

        # In MLX mode, we use MLX tokenizers. In PyTorch mode, we use HF tokenizer.
        self.tokenizer = None
        self.baseline_tokenizer = None
        self.fine_tuned_tokenizer = None

        if self.use_mlx:
            try:
                from mlx_lm import load as mlx_load  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "MLX validation requested but `mlx-lm` is not importable.\n"
                    "Fix:\n"
                    "  - `uv pip install mlx mlx-lm`\n"
                    f"Error: {type(e).__name__}: {e}"
                )

            # Pass tokenizer_config to follow the recommended fix for the "mistral regex" warning when applicable.
            tok_cfg = {"fix_mistral_regex": True}

            logger.info(f"Loading baseline MLX model from {self.baseline_mlx_path}")
            self.baseline_model, self.baseline_tokenizer = mlx_load(self.baseline_mlx_path, tokenizer_config=tok_cfg)

            logger.info(f"Loading fine-tuned MLX model from {self.fine_tuned_mlx_path}")
            try:
                self.fine_tuned_model, self.fine_tuned_tokenizer = mlx_load(self.fine_tuned_mlx_path, tokenizer_config=tok_cfg)
            except Exception as e:
                # Common failure mode: an older/broken checkpoint MLX export has weights/config mismatch.
                # Example: "Received XXX parameters not in model: model.embed_tokens.biases, model.embed_tokens.scales"
                msg = str(e)
                should_repair = ("parameters not in model" in msg) or ("Received " in msg and "not in model" in msg)
                if not should_repair:
                    raise

                ckpt_dir = Path(self.fine_tuned_path)
                # Try to find HF checkpoint dir (should contain config.json + model.safetensors or shards).
                hf_dir = ckpt_dir
                if not (hf_dir / "config.json").exists():
                    # If user passed an MLX dir directly, try its parent as HF checkpoint dir.
                    hf_dir = Path(self.fine_tuned_mlx_path).parent

                repaired_dir = ckpt_dir / "mlx_model_repaired"
                logger.warning(
                    "Fine-tuned MLX model failed to load due to config/weights mismatch. "
                    f"Rebuilding MLX export from HF checkpoint into: {repaired_dir}"
                )
                self._rebuild_mlx_from_hf_checkpoint(hf_dir=hf_dir, mlx_out_dir=repaired_dir)
                self.fine_tuned_model, self.fine_tuned_tokenizer = mlx_load(str(repaired_dir), tokenizer_config=tok_cfg)
        else:
            # Pick device explicitly (avoid device_map="auto" on MPS which can create meta/placeholder tensors).
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

            # Load tokenizers
            logger.info(f"Loading tokenizer from {base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Prefer bfloat16 on MPS when available; otherwise use float16 on CUDA, float32 on CPU.
            if self.device.type == "mps":
                torch_dtype = torch.bfloat16
            elif self.device.type == "cuda":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

            # Load baseline model
            logger.info(f"Loading baseline model: {base_model}")
            self.baseline_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=False,  # avoid meta tensors; more reliable on MPS/CPU
            )
            self.baseline_model.to(self.device)
            self.baseline_model.eval()

            # Load fine-tuned model
            logger.info(f"Loading fine-tuned model from {fine_tuned_path}")
            self.fine_tuned_model = AutoModelForCausalLM.from_pretrained(
                fine_tuned_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=False,
            )
            self.fine_tuned_model.to(self.device)
            self.fine_tuned_model.eval()
        
        # Initialize teacher for scoring
        self.teacher = self._init_teacher()
    
    def _init_teacher(self):
        """Initialize teacher model for scoring"""
        try:
            from scripts.training.train_rlaif import TeacherModel
            api_key_env = "OPENAI_API_KEY" if self.teacher_provider == "openai" else "ANTHROPIC_API_KEY"
            return TeacherModel(self.teacher_provider, self.teacher_model, api_key_env)
        except Exception as e:
            logger.warning(f"Could not initialize teacher model: {e}")
            return None
    
    def generate_code(self, model, prompt: str, language: str, max_tokens: int = 512, *, tokenizer=None) -> Dict[str, float | str]:
        """Generate code from a model and return code + throughput stats.

        Returns:
            {
              "code": str,
              "seconds": float,
              "output_tokens": int,
              "tokens_per_sec": float,
            }
        """
        formatted_prompt = f"Write high-quality {language} code:\n\n{prompt}\n\nCode:"

        if self.use_mlx:
            try:
                from mlx_lm import generate as mlx_generate  # type: ignore
            except Exception as e:
                raise RuntimeError(f"mlx_lm.generate import failed: {type(e).__name__}: {e}")

            tok = tokenizer
            if tok is None:
                tok = self.baseline_tokenizer or self.fine_tuned_tokenizer

            sampler = None
            try:
                from mlx_lm.sample_utils import make_sampler  # type: ignore
                sampler = make_sampler(temp=0.8, top_p=0.95, top_k=50)
            except Exception:
                sampler = None

            t0 = time.time()
            text = mlx_generate(
                model,
                tok,
                prompt=formatted_prompt,
                max_tokens=int(max_tokens),
                sampler=sampler,
            )
            dt = max(1e-9, time.time() - t0)
            if isinstance(text, str) and text.startswith(formatted_prompt):
                code = text[len(formatted_prompt):].strip()
            else:
                code = str(text)

            out_tokens = 0
            try:
                enc = getattr(tok, "encode", None)
                if callable(enc):
                    ids = enc(code)
                    if isinstance(ids, list):
                        out_tokens = len(ids)
            except Exception:
                out_tokens = 0

            tps = float(out_tokens) / float(dt) if dt > 0 else 0.0
            return {"code": code, "seconds": float(dt), "output_tokens": int(out_tokens), "tokens_per_sec": float(tps)}

        assert self.tokenizer is not None
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        # Ensure inputs are on the same device as the model.
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            t0 = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            dt = max(1e-9, time.time() - t0)

        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Approx generated token count
        out_tokens = int(outputs.shape[1] - inputs["input_ids"].shape[1]) if hasattr(outputs, "shape") else 0
        tps = float(out_tokens) / float(dt) if dt > 0 else 0.0
        return {"code": generated_text, "seconds": float(dt), "output_tokens": int(out_tokens), "tokens_per_sec": float(tps)}

    def _rebuild_mlx_from_hf_checkpoint(self, *, hf_dir: Path, mlx_out_dir: Path) -> None:
        """Rebuild an MLX export from a local HF checkpoint directory using mlx_lm.convert.

        This fixes older/broken checkpoint MLX exports where config.json doesn't match weights.
        """
        hf_dir = Path(hf_dir)
        mlx_out_dir = Path(mlx_out_dir)
        if not hf_dir.exists():
            raise FileNotFoundError(f"HF checkpoint dir not found: {hf_dir}")

        # Align export quantization with baseline MLX model path to keep comparisons apples-to-apples.
        base_l = str(self.baseline_mlx_path).lower()
        quant = None
        if "/q4" in base_l or "q4_bit" in base_l:
            quant = ("-q", "--q-bits", "4")
        elif "/q8" in base_l or "q8_bit" in base_l:
            quant = ("-q", "--q-bits", "8")

        # mlx_lm.convert requires output dir not to exist.
        try:
            if mlx_out_dir.exists():
                import shutil
                shutil.rmtree(mlx_out_dir)
        except Exception:
            pass

        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            str(hf_dir),
            "--mlx-path",
            str(mlx_out_dir),
        ]
        if quant:
            cmd += list(quant)

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            raise RuntimeError(f"mlx_lm.convert failed (code={proc.returncode}).\nstdout:\n{out}\nstderr:\n{err}")

        # Minimal sanity check
        if not (mlx_out_dir / "config.json").exists():
            raise RuntimeError(f"Rebuilt MLX dir missing config.json: {mlx_out_dir}")
    
    def score_code(self, code: str, prompt: str, language: str) -> float:
        """Score code using teacher model"""
        if self.teacher:
            return self.teacher.score_code(code, prompt, language)
        else:
            # Fallback: simple heuristic scoring
            score = 0.5
            if len(code) > 50:
                score += 0.1
            if "def " in code or "class " in code or "fn " in code:
                score += 0.2
            if "\n" in code:
                score += 0.1
            return min(1.0, score)
    
    def validate(self, test_prompts: List[Dict]) -> Dict:
        """Validate models on test prompts"""
        results = {
            'baseline_scores': [],
            'fine_tuned_scores': [],
            'improvements': [],
            'examples': []
        }
        
        logger.info(f"Validating on {len(test_prompts)} test prompts...")
        
        for i, test_case in enumerate(test_prompts):
            prompt = test_case['prompt']
            language = test_case.get('language', 'python')
            
            logger.info(f"\n[{i+1}/{len(test_prompts)}] Testing: {prompt[:50]}...")
            
            # Generate from baseline
            baseline_gen = self.generate_code(
                self.baseline_model,
                prompt,
                language,
                tokenizer=self.baseline_tokenizer,
            )
            baseline_code = str(baseline_gen.get("code", "") or "")
            baseline_score = self.score_code(baseline_code, prompt, language)
            
            # Generate from fine-tuned
            fine_tuned_gen = self.generate_code(
                self.fine_tuned_model,
                prompt,
                language,
                tokenizer=self.fine_tuned_tokenizer,
            )
            fine_tuned_code = str(fine_tuned_gen.get("code", "") or "")
            fine_tuned_score = self.score_code(fine_tuned_code, prompt, language)
            
            improvement = fine_tuned_score - baseline_score
            
            results['baseline_scores'].append(baseline_score)
            results['fine_tuned_scores'].append(fine_tuned_score)
            results['improvements'].append(improvement)
            
            # Store example
            results['examples'].append({
                'prompt': prompt,
                'language': language,
                'baseline_code': baseline_code,
                'fine_tuned_code': fine_tuned_code,
                'baseline_score': baseline_score,
                'fine_tuned_score': fine_tuned_score,
                'improvement': improvement,
                'baseline_gen_seconds': float(baseline_gen.get("seconds", 0.0) or 0.0),
                'baseline_gen_tokens': int(baseline_gen.get("output_tokens", 0) or 0),
                'baseline_gen_tps': float(baseline_gen.get("tokens_per_sec", 0.0) or 0.0),
                'fine_tuned_gen_seconds': float(fine_tuned_gen.get("seconds", 0.0) or 0.0),
                'fine_tuned_gen_tokens': int(fine_tuned_gen.get("output_tokens", 0) or 0),
                'fine_tuned_gen_tps': float(fine_tuned_gen.get("tokens_per_sec", 0.0) or 0.0),
            })
            
            logger.info(f"  Baseline Score: {baseline_score:.4f}")
            logger.info(f"  Fine-tuned Score: {fine_tuned_score:.4f}")
            logger.info(f"  Improvement: {improvement:+.4f}")
            logger.info(
                "  Generation (tok/s): "
                f"baseline={float(baseline_gen.get('tokens_per_sec', 0.0) or 0.0):.1f} "
                f"({int(baseline_gen.get('output_tokens', 0) or 0)} tok / {float(baseline_gen.get('seconds', 0.0) or 0.0):.2f}s), "
                f"fine_tuned={float(fine_tuned_gen.get('tokens_per_sec', 0.0) or 0.0):.1f} "
                f"({int(fine_tuned_gen.get('output_tokens', 0) or 0)} tok / {float(fine_tuned_gen.get('seconds', 0.0) or 0.0):.2f}s)"
            )
        
        # Calculate statistics
        results['stats'] = {
            'avg_baseline_score': sum(results['baseline_scores']) / len(results['baseline_scores']),
            'avg_fine_tuned_score': sum(results['fine_tuned_scores']) / len(results['fine_tuned_scores']),
            'avg_improvement': sum(results['improvements']) / len(results['improvements']),
            'num_improved': sum(1 for imp in results['improvements'] if imp > 0),
            'num_degraded': sum(1 for imp in results['improvements'] if imp < 0),
            'num_equal': sum(1 for imp in results['improvements'] if imp == 0),
        }
        
        return results
    
    def print_report(self, results: Dict):
        """Print validation report"""
        stats = results['stats']
        
        print("\n" + "="*80)
        print("VALIDATION REPORT: Pre-Training(baseline model) vs Post-Training (fine tuned model)")
        print("="*80)
        print(f"\nTest Cases: {len(results['baseline_scores'])}")
        print(f"\nAverage Reward Scores:")
        print(f"  Baseline Model:    {stats['avg_baseline_score']:.4f}")
        print(f"  Fine-tuned Model:  {stats['avg_fine_tuned_score']:.4f}")
        print(f"  Average Improvement: {stats['avg_improvement']:+.4f} ({stats['avg_improvement']/stats['avg_baseline_score']*100:+.2f}%)")
        print(f"\nImprovement Distribution:")
        print(f"  Improved:  {stats['num_improved']} cases")
        print(f"  Degraded:  {stats['num_degraded']} cases")
        print(f"  Equal:     {stats['num_equal']} cases")
        
        print("\n" + "="*80)
        print("EXAMPLE COMPARISONS")
        print("="*80)
        
        # Show top 3 improvements
        sorted_examples = sorted(results['examples'], key=lambda x: x['improvement'], reverse=True)
        for i, example in enumerate(sorted_examples[:3], 1):
            print(f"\n[Example {i}] {example['prompt'][:60]}...")
            print(f"Language: {example['language']}")
            print(f"Baseline Score: {example['baseline_score']:.4f}")
            print(f"Fine-tuned Score: {example['fine_tuned_score']:.4f}")
            print(f"Improvement: {example['improvement']:+.4f}")
            try:
                print(
                    "Generation (tok/s): "
                    f"baseline={float(example.get('baseline_gen_tps', 0.0)):.1f}, "
                    f"fine_tuned={float(example.get('fine_tuned_gen_tps', 0.0)):.1f}"
                )
            except Exception:
                pass
            print(f"\nBaseline Code:")
            print("-" * 40)
            print(_truncate(str(example.get('baseline_code', '') or ''), self._print_chars))
            print(f"\nFine-tuned Code:")
            print("-" * 40)
            print(_truncate(str(example.get('fine_tuned_code', '') or ''), self._print_chars))
            print()


def load_test_prompts(file_path: str) -> List[Dict]:
    """Load test prompts from file"""
    if not os.path.exists(file_path):
        # Return default test prompts
        return [
            {"prompt": "Implement a binary search function", "language": "python"},
            {"prompt": "Create a thread-safe queue", "language": "cpp"},
            {"prompt": "Write a function to parse JSON safely", "language": "rust"},
            {"prompt": "Implement a decorator that measures execution time", "language": "python"},
            {"prompt": "Create a RAII wrapper for file handling", "language": "cpp"},
        ]
    
    prompts = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Validate fine-tuned model quality")
    parser.add_argument(
        '--base_model',
        type=str,
        default='Qwen/Qwen2.5-3B-Instruct',
        help='Base model name'
    )
    parser.add_argument(
        '--fine_tuned_path',
        type=str,
        required=True,
        help='Path to fine-tuned model checkpoint'
    )
    parser.add_argument(
        '--test_prompts',
        type=str,
        default='./data/eval.jsonl',
        help='Path to test prompts file (JSONL format)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=20,
        help='Number of test prompts to evaluate (default: 5). Use 0 to evaluate all prompts.'
    )
    parser.add_argument(
        '--print_chars',
        type=int,
        default=500,
        help='Max characters of code to print per sample (default: 500). Use 0 to print full code.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./validation_results.json',
        help='Output file for validation results'
    )
    parser.add_argument(
        '--teacher_provider',
        type=str,
        default='openai',
        choices=['openai', 'anthropic'],
        help='Teacher model provider'
    )
    parser.add_argument(
        '--teacher_model',
        type=str,
        default=None,
        help="Teacher model name (defaults to provider-appropriate model: Anthropic→claude-3-5-haiku-20241022, OpenAI→gpt-4-turbo-preview)"
    )
    parser.add_argument(
        '--baseline_mlx_path',
        type=str,
        default='./mlx_model/q4',
        help='Path to baseline MLX model directory (e.g., ./mlx_model/q4)'
    )
    parser.add_argument(
        '--fine_tuned_mlx_path',
        type=str,
        default=None,
        help='Path to fine-tuned MLX model directory (defaults to <fine_tuned_path>/mlx_model if present)'
    )
    # Python 3.10+: allows `--use_mlx` / `--no-use_mlx`
    parser.add_argument(
        '--use_mlx',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use MLX for generation for both baseline and fine-tuned models (recommended on Apple Silicon).'
    )
    
    args = parser.parse_args()
    
    # Load test prompts
    test_prompts = load_test_prompts(args.test_prompts)
    if int(args.max_samples) > 0:
        test_prompts = test_prompts[: int(args.max_samples)]
    logger.info(f"Loaded {len(test_prompts)} test prompts")
    
    # Initialize validator
    validator = ModelValidator(
        base_model=args.base_model,
        fine_tuned_path=args.fine_tuned_path,
        teacher_provider=args.teacher_provider,
        teacher_model=args.teacher_model,
        use_mlx=bool(args.use_mlx),
        baseline_mlx_path=args.baseline_mlx_path,
        fine_tuned_mlx_path=args.fine_tuned_mlx_path,
        print_chars=int(args.print_chars),
    )
    
    # Run validation
    results = validator.validate(test_prompts)
    
    # Print report
    validator.print_report(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    results['metadata'] = {
        'base_model': args.base_model,
        'fine_tuned_path': args.fine_tuned_path,
        'use_mlx': bool(args.use_mlx),
        'baseline_mlx_path': args.baseline_mlx_path if bool(args.use_mlx) else None,
        'fine_tuned_mlx_path': args.fine_tuned_mlx_path if bool(args.use_mlx) else None,
        'test_prompts_file': args.test_prompts,
        'num_test_cases': len(test_prompts),
        'validation_date': datetime.now().isoformat(),
        'teacher_provider': args.teacher_provider,
        'teacher_model': args.teacher_model
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nValidation results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()

