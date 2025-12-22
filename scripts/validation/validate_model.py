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
import random
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.tensorboard import SummaryWriter

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
        generation_temperature: float = 0.2,
        generation_top_p: float = 0.95,
        generation_top_k: int = 50,
        generation_seed: int | None = 0,
        allow_no_teacher: bool = False,
        score_mode: str = "absolute",
        tensorboard_dir: str | None = None,
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

        self.generation_temperature = float(generation_temperature)
        self.generation_top_p = float(generation_top_p)
        self.generation_top_k = int(generation_top_k)
        self.generation_seed = None if generation_seed is None else int(generation_seed)
        self._seed_everything(self.generation_seed)
        self.allow_no_teacher = bool(allow_no_teacher)
        self.score_mode = str(score_mode or "absolute").strip().lower()
        if self.score_mode not in {"absolute", "normalized", "both"}:
            raise ValueError("score_mode must be one of: 'absolute', 'normalized', 'both'")

        # Cache teacher reference per prompt+language to reduce API cost when computing normalized reward.
        # key: f"{prompt}:{language}" -> (teacher_code, teacher_score)
        self._teacher_ref_cache: dict[str, tuple[str, float]] = {}

        # Optional TensorBoard logging for validation runs
        self._tb = None
        tb_dir = str(tensorboard_dir) if tensorboard_dir else None
        if tb_dir:
            try:
                Path(tb_dir).mkdir(parents=True, exist_ok=True)
                self._tb = SummaryWriter(tb_dir)
            except Exception:
                self._tb = None

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
        if self.teacher is None and not self.allow_no_teacher:
            raise RuntimeError(
                "Teacher model could not be initialized (missing API key or dependency). "
                "Set ANTHROPIC_API_KEY / OPENAI_API_KEY, or pass --allow_no_teacher to use heuristic scoring."
            )

    def _seed_everything(self, seed: int | None) -> None:
        """Best-effort seeding for reproducible generation."""
        if seed is None:
            return
        try:
            random.seed(seed)
        except Exception:
            pass
        try:
            torch.manual_seed(seed)
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        # MLX seeding (best-effort; only when available)
        try:
            import mlx.core as mx  # type: ignore
            mx.random.seed(seed)
        except Exception:
            pass
    
    def _init_teacher(self):
        """Initialize teacher model for scoring"""
        try:
            from scripts.training.train_rlaif import TeacherModel
            api_key_env = "OPENAI_API_KEY" if self.teacher_provider == "openai" else "ANTHROPIC_API_KEY"
            return TeacherModel(self.teacher_provider, self.teacher_model, api_key_env)
        except Exception as e:
            logger.warning(f"Could not initialize teacher model: {e}")
            return None
    
    def generate_code(
        self,
        model,
        prompt: str,
        language: str,
        max_tokens: int = 512,
        *,
        tokenizer=None,
        seed: int | None = None,
    ) -> Dict[str, float | str]:
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

            self._seed_everything(seed if seed is not None else self.generation_seed)
            sampler = None
            try:
                from mlx_lm.sample_utils import make_sampler  # type: ignore
                # For reproducibility/less noise in eval, default to lower temperature.
                # If temp <= 0, keep sampler None (mlx_lm.generate typically falls back to greedy).
                if float(self.generation_temperature) > 0:
                    sampler = make_sampler(
                        temp=float(self.generation_temperature),
                        top_p=float(self.generation_top_p),
                        top_k=int(self.generation_top_k),
                    )
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
        self._seed_everything(seed if seed is not None else self.generation_seed)
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
                temperature=float(self.generation_temperature),
                top_k=int(self.generation_top_k),
                top_p=float(self.generation_top_p),
                do_sample=bool(float(self.generation_temperature) > 0),
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

    def prompt_difficulty(self, prompt: str, language: str) -> dict[str, float]:
        """Compute a rubric-aligned prompt difficulty index.

        Components:
        - length proxy: token_len (formatted prompt) fallback to char_len
        - rubric demand: Correctness / Code Quality / Efficiency / Documentation (0..1 each)
        - composite index: length_proxy * lang_weight * (1 + 0.75 * rubric_demand)
        """
        p = str(prompt or "")
        lg = str(language or "python").lower()
        formatted_prompt = f"Write high-quality {lg} code:\n\n{p}\n\nCode:"

        def _rubric_components(text: str, lang: str) -> dict[str, float]:
            t = (text or "").lower()

            def count_any(words: list[str]) -> int:
                return sum(1 for w in words if w in t)

            # Correctness
            corr_hits = 0
            corr_hits += count_any(
                [
                    "edge case",
                    "corner case",
                    "validate",
                    "invalid",
                    "error handling",
                    "robust",
                    "safely",
                    "thread-safe",
                    "thread safe",
                    "lock-free",
                    "deadlock",
                    "race",
                    "atomic",
                    "parse",
                    "parser",
                    "serialize",
                    "deserialize",
                    "json",
                    "unicode",
                    "overflow",
                    "underflow",
                    "null",
                    "nullptr",
                ]
            )
            if re.search(r"\bconstraints?\b|\bguarantee\b|\bmust\b|\bshould\b", t):
                corr_hits += 1
            correctness = min(1.0, corr_hits / 6.0)

            # Code quality
            qual_hits = 0
            qual_hits += count_any(
                [
                    "clean",
                    "readable",
                    "well-structured",
                    "well structured",
                    "maintainable",
                    "refactor",
                    "design pattern",
                    "singleton",
                    "raii",
                    "interface",
                    "abstraction",
                    "encapsulation",
                    "modular",
                    "unit test",
                    "tests",
                ]
            )
            if any(w in t for w in ["class ", "api", "library", "module"]):
                qual_hits += 1
            code_quality = min(1.0, qual_hits / 6.0)

            # Efficiency
            eff_hits = 0
            eff_hits += count_any(
                [
                    "efficient",
                    "optimize",
                    "performance",
                    "fast",
                    "low latency",
                    "high throughput",
                    "big-o",
                    "o(",
                    "time complexity",
                    "space complexity",
                    "memory",
                    "constant time",
                    "log n",
                    "n log n",
                    "linear time",
                ]
            )
            if re.search(r"\b\d+\s*(ms|seconds|s)\b", t):
                eff_hits += 1
            if re.search(r"\b\d+\s*(items|elements|rows|cols|nodes)\b|\bup to\b", t):
                eff_hits += 1
            efficiency = min(1.0, eff_hits / 6.0)

            # Documentation
            doc_hits = 0
            doc_hits += count_any(
                [
                    "document",
                    "documentation",
                    "docstring",
                    "comments",
                    "commented",
                    "well-documented",
                    "well documented",
                    "explain",
                    "explanation",
                    "examples",
                ]
            )
            documentation = min(1.0, doc_hits / 4.0)

            rubric_demand = (0.3 * correctness) + (0.3 * code_quality) + (0.2 * efficiency) + (0.2 * documentation)

            if lang in ("cpp", "c++"):
                lang_weight = 1.10
            elif lang == "rust":
                lang_weight = 1.15
            else:
                lang_weight = 1.00

            return {
                "correctness": float(correctness),
                "code_quality": float(code_quality),
                "efficiency": float(efficiency),
                "documentation": float(documentation),
                "rubric_demand": float(min(1.0, max(0.0, rubric_demand))),
                "lang_weight": float(lang_weight),
            }

        tok_len = 0.0
        try:
            if self.use_mlx:
                tok = self.baseline_tokenizer or self.fine_tuned_tokenizer
                enc = getattr(tok, "encode", None) if tok is not None else None
                if callable(enc):
                    ids = enc(formatted_prompt)
                    if isinstance(ids, list):
                        tok_len = float(len(ids))
            else:
                if self.tokenizer is not None:
                    ids = self.tokenizer.encode(formatted_prompt)
                    tok_len = float(len(ids))
        except Exception:
            tok_len = 0.0

        char_len = float(len(p))
        comps = _rubric_components(p, lg)
        length_proxy = float(tok_len if tok_len > 0 else char_len)
        idx = float(length_proxy) * float(comps["lang_weight"]) * (1.0 + 0.75 * float(comps["rubric_demand"]))
        return {
            "token_len": float(tok_len),
            "char_len": float(char_len),
            "lang_weight": float(comps["lang_weight"]),
            "rubric_demand": float(comps["rubric_demand"]),
            "demand_correctness": float(comps["correctness"]),
            "demand_code_quality": float(comps["code_quality"]),
            "demand_efficiency": float(comps["efficiency"]),
            "demand_documentation": float(comps["documentation"]),
            "index": float(idx),
        }

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

    def _get_teacher_reference(self, prompt: str, language: str) -> tuple[str, float]:
        """Get (teacher_code, teacher_score) for normalized reward comparisons."""
        if not self.teacher:
            raise RuntimeError("Teacher is not initialized; cannot compute normalized reward.")
        key = f"{prompt}:{language}"
        if key in self._teacher_ref_cache:
            return self._teacher_ref_cache[key]
        teacher_code = self.teacher.generate(prompt, language)
        teacher_score = float(self.teacher.score_code(teacher_code, prompt, language))
        self._teacher_ref_cache[key] = (teacher_code, teacher_score)
        return teacher_code, teacher_score

    def score_student(
        self,
        code: str,
        prompt: str,
        language: str,
        *,
        eps: float = 1e-6,
    ) -> dict[str, float]:
        """Return scoring dict containing absolute score and (optionally) normalized reward."""
        student_score = float(self.score_code(code, prompt, language))
        out: dict[str, float] = {"student_score": student_score}
        if self.score_mode in {"normalized", "both"}:
            teacher_code, teacher_score = self._get_teacher_reference(prompt, language)
            # Match training: reward = student_score / (teacher_score + eps)
            reward = float(student_score) / (float(teacher_score) + float(eps))
            out["teacher_score"] = float(teacher_score)
            out["reward_normalized"] = float(reward)
        return out
    
    def validate(
        self,
        test_prompts: List[Dict],
        *,
        num_generations_per_prompt: int = 1,
        aggregate: str = "mean",
    ) -> Dict:
        """Validate models on test prompts"""
        results = {
            # Absolute teacher scores (0..1)
            'baseline_scores': [],
            'fine_tuned_scores': [],
            'improvements': [],
            # Normalized rewards (student_score / teacher_score) if enabled
            'baseline_rewards': [],
            'fine_tuned_rewards': [],
            'reward_improvements': [],
            'examples': []
        }
        
        logger.info(f"Validating on {len(test_prompts)} test prompts...")
        
        for i, test_case in enumerate(test_prompts):
            prompt = test_case['prompt']
            language = test_case.get('language', 'python')
            diff = self.prompt_difficulty(prompt, language)
            
            logger.info(f"\n[{i+1}/{len(test_prompts)}] Testing: {prompt[:50]}...")
            try:
                if self._tb:
                    self._tb.add_scalar("Validation/PromptDifficulty/TokenLen", float(diff.get("token_len", 0.0)), i)
                    self._tb.add_scalar("Validation/PromptDifficulty/CharLen", float(diff.get("char_len", 0.0)), i)
                    self._tb.add_scalar("Validation/PromptDifficulty/Index", float(diff.get("index", 0.0)), i)
                    self._tb.add_scalar("Validation/PromptDifficulty/RubricDemand", float(diff.get("rubric_demand", 0.0)), i)
                    self._tb.add_scalar("Validation/PromptDifficulty/Demand_Correctness", float(diff.get("demand_correctness", 0.0)), i)
                    self._tb.add_scalar("Validation/PromptDifficulty/Demand_CodeQuality", float(diff.get("demand_code_quality", 0.0)), i)
                    self._tb.add_scalar("Validation/PromptDifficulty/Demand_Efficiency", float(diff.get("demand_efficiency", 0.0)), i)
                    self._tb.add_scalar("Validation/PromptDifficulty/Demand_Documentation", float(diff.get("demand_documentation", 0.0)), i)
            except Exception:
                pass
            
            n = max(1, int(num_generations_per_prompt))
            agg = str(aggregate or "mean").strip().lower()
            if agg not in {"mean", "max"}:
                raise ValueError(f"Unsupported aggregate={aggregate!r}. Use 'mean' or 'max'.")

            baseline_gens = []
            fine_tuned_gens = []
            baseline_scores = []
            fine_tuned_scores = []
            baseline_rewards = []
            fine_tuned_rewards = []

            # Use deterministic per-sample seed offsets to make paired comparisons stable.
            base_seed = self.generation_seed
            for j in range(n):
                s = None if base_seed is None else int(base_seed) + int(i) * 1000 + int(j)
                baseline_gen = self.generate_code(
                    self.baseline_model,
                    prompt,
                    language,
                    tokenizer=self.baseline_tokenizer,
                    seed=s,
                )
                baseline_code = str(baseline_gen.get("code", "") or "")
                baseline_sc = self.score_student(baseline_code, prompt, language)
                baseline_score = float(baseline_sc.get("student_score", 0.0))
                baseline_reward = float(baseline_sc.get("reward_normalized", 0.0))
                baseline_gens.append(baseline_gen)
                baseline_scores.append(baseline_score)
                if self.score_mode in {"normalized", "both"}:
                    baseline_rewards.append(baseline_reward)

                fine_tuned_gen = self.generate_code(
                    self.fine_tuned_model,
                    prompt,
                    language,
                    tokenizer=self.fine_tuned_tokenizer,
                    seed=s,
                )
                fine_tuned_code = str(fine_tuned_gen.get("code", "") or "")
                fine_tuned_sc = self.score_student(fine_tuned_code, prompt, language)
                fine_tuned_score = float(fine_tuned_sc.get("student_score", 0.0))
                fine_tuned_reward = float(fine_tuned_sc.get("reward_normalized", 0.0))
                fine_tuned_gens.append(fine_tuned_gen)
                fine_tuned_scores.append(fine_tuned_score)
                if self.score_mode in {"normalized", "both"}:
                    fine_tuned_rewards.append(fine_tuned_reward)

            if agg == "max":
                baseline_score = max(baseline_scores)
                fine_tuned_score = max(fine_tuned_scores)
                baseline_pick = int(baseline_scores.index(baseline_score))
                fine_tuned_pick = int(fine_tuned_scores.index(fine_tuned_score))
            else:
                baseline_score = sum(baseline_scores) / len(baseline_scores)
                fine_tuned_score = sum(fine_tuned_scores) / len(fine_tuned_scores)
                baseline_pick = 0
                fine_tuned_pick = 0

            # Aggregate normalized reward similarly (if enabled)
            baseline_reward_agg = None
            fine_tuned_reward_agg = None
            if self.score_mode in {"normalized", "both"}:
                if baseline_rewards and fine_tuned_rewards:
                    if agg == "max":
                        baseline_reward_agg = max(baseline_rewards)
                        fine_tuned_reward_agg = max(fine_tuned_rewards)
                    else:
                        baseline_reward_agg = sum(baseline_rewards) / len(baseline_rewards)
                        fine_tuned_reward_agg = sum(fine_tuned_rewards) / len(fine_tuned_rewards)

            baseline_code = str(baseline_gens[baseline_pick].get("code", "") or "")
            fine_tuned_code = str(fine_tuned_gens[fine_tuned_pick].get("code", "") or "")
            
            improvement = fine_tuned_score - baseline_score
            reward_improvement = None
            if baseline_reward_agg is not None and fine_tuned_reward_agg is not None:
                reward_improvement = float(fine_tuned_reward_agg) - float(baseline_reward_agg)
            
            results['baseline_scores'].append(baseline_score)
            results['fine_tuned_scores'].append(fine_tuned_score)
            results['improvements'].append(improvement)
            if baseline_reward_agg is not None and fine_tuned_reward_agg is not None:
                results['baseline_rewards'].append(float(baseline_reward_agg))
                results['fine_tuned_rewards'].append(float(fine_tuned_reward_agg))
                results['reward_improvements'].append(float(reward_improvement or 0.0))
            
            # Store example
            results['examples'].append({
                'prompt': prompt,
                'language': language,
                'prompt_difficulty': diff,
                'baseline_code': baseline_code,
                'fine_tuned_code': fine_tuned_code,
                'baseline_score': baseline_score,
                'fine_tuned_score': fine_tuned_score,
                'improvement': improvement,
                'baseline_reward_normalized': float(baseline_reward_agg) if baseline_reward_agg is not None else None,
                'fine_tuned_reward_normalized': float(fine_tuned_reward_agg) if fine_tuned_reward_agg is not None else None,
                'reward_improvement': float(reward_improvement) if reward_improvement is not None else None,
                'num_generations_per_prompt': n,
                'aggregate': agg,
                'baseline_score_samples': baseline_scores,
                'fine_tuned_score_samples': fine_tuned_scores,
                'baseline_reward_samples': baseline_rewards if (self.score_mode in {"normalized", "both"}) else None,
                'fine_tuned_reward_samples': fine_tuned_rewards if (self.score_mode in {"normalized", "both"}) else None,
                'baseline_gen_seconds': float(baseline_gens[baseline_pick].get("seconds", 0.0) or 0.0),
                'baseline_gen_tokens': int(baseline_gens[baseline_pick].get("output_tokens", 0) or 0),
                'baseline_gen_tps': float(baseline_gens[baseline_pick].get("tokens_per_sec", 0.0) or 0.0),
                'fine_tuned_gen_seconds': float(fine_tuned_gens[fine_tuned_pick].get("seconds", 0.0) or 0.0),
                'fine_tuned_gen_tokens': int(fine_tuned_gens[fine_tuned_pick].get("output_tokens", 0) or 0),
                'fine_tuned_gen_tps': float(fine_tuned_gens[fine_tuned_pick].get("tokens_per_sec", 0.0) or 0.0),
            })
            
            logger.info(f"  Baseline Score: {baseline_score:.4f}")
            logger.info(f"  Fine-tuned Score: {fine_tuned_score:.4f}")
            logger.info(f"  Improvement: {improvement:+.4f}")
            if baseline_reward_agg is not None and fine_tuned_reward_agg is not None:
                logger.info(f"  Baseline Reward(norm): {float(baseline_reward_agg):.4f}")
                logger.info(f"  Fine-tuned Reward(norm): {float(fine_tuned_reward_agg):.4f}")
                logger.info(f"  Reward Improvement: {float(reward_improvement or 0.0):+.4f}")
            logger.info(
                "  Generation (tok/s): "
                f"baseline={float(baseline_gens[baseline_pick].get('tokens_per_sec', 0.0) or 0.0):.1f} "
                f"({int(baseline_gens[baseline_pick].get('output_tokens', 0) or 0)} tok / {float(baseline_gens[baseline_pick].get('seconds', 0.0) or 0.0):.2f}s), "
                f"fine_tuned={float(fine_tuned_gens[fine_tuned_pick].get('tokens_per_sec', 0.0) or 0.0):.1f} "
                f"({int(fine_tuned_gens[fine_tuned_pick].get('output_tokens', 0) or 0)} tok / {float(fine_tuned_gens[fine_tuned_pick].get('seconds', 0.0) or 0.0):.2f}s)"
            )

            # Validation TB (per prompt)
            try:
                if self._tb:
                    self._tb.add_scalar("Validation/Baseline/Score", float(baseline_score), i)
                    self._tb.add_scalar("Validation/FineTuned/Score", float(fine_tuned_score), i)
                    self._tb.add_scalar("Validation/Improvement/Score", float(improvement), i)
                    if baseline_reward_agg is not None and fine_tuned_reward_agg is not None:
                        self._tb.add_scalar("Validation/Baseline/Reward_Normalized", float(baseline_reward_agg), i)
                        self._tb.add_scalar("Validation/FineTuned/Reward_Normalized", float(fine_tuned_reward_agg), i)
                        self._tb.add_scalar("Validation/Improvement/Reward_Normalized", float(reward_improvement or 0.0), i)
            except Exception:
                pass
        
        # Calculate statistics
        stats = {
            'avg_baseline_score': sum(results['baseline_scores']) / len(results['baseline_scores']),
            'avg_fine_tuned_score': sum(results['fine_tuned_scores']) / len(results['fine_tuned_scores']),
            'avg_improvement': sum(results['improvements']) / len(results['improvements']),
            'num_improved': sum(1 for imp in results['improvements'] if imp > 0),
            'num_degraded': sum(1 for imp in results['improvements'] if imp < 0),
            'num_equal': sum(1 for imp in results['improvements'] if imp == 0),
        }
        if self.score_mode in {"normalized", "both"} and results.get("baseline_rewards"):
            stats.update({
                "avg_baseline_reward_normalized": sum(results["baseline_rewards"]) / len(results["baseline_rewards"]),
                "avg_fine_tuned_reward_normalized": sum(results["fine_tuned_rewards"]) / len(results["fine_tuned_rewards"]),
                "avg_reward_improvement": sum(results["reward_improvements"]) / len(results["reward_improvements"]),
            })
        results["stats"] = stats
        try:
            if self._tb:
                self._tb.add_scalar("Validation/Summary/Avg_Baseline_Score", float(stats.get("avg_baseline_score", 0.0)), 0)
                self._tb.add_scalar("Validation/Summary/Avg_FineTuned_Score", float(stats.get("avg_fine_tuned_score", 0.0)), 0)
                if "avg_fine_tuned_reward_normalized" in stats:
                    self._tb.add_scalar("Validation/Summary/Avg_FineTuned_Reward_Normalized", float(stats.get("avg_fine_tuned_reward_normalized", 0.0)), 0)
                self._tb.flush()
        except Exception:
            pass
        
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


def dedupe_test_prompts(prompts: List[Dict]) -> List[Dict]:
    """De-duplicate prompts by (prompt, language). Keeps first occurrence."""
    out: List[Dict] = []
    seen = set()
    for p in prompts:
        try:
            key = (str(p.get("prompt", "") or ""), str(p.get("language", "python") or "python"))
        except Exception:
            key = (str(p), "python")
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser(description="Validate fine-tuned model quality")
    parser.add_argument(
        '--base_model',
        type=str,
        default='Qwen/Qwen2.5-Coder-3B-Instruct',
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
        default='anthropic',
        choices=['openai', 'anthropic'],
        help='Teacher model provider'
    )
    parser.add_argument(
        '--teacher_model',
        type=str,
        default='claude-3-5-haiku-20241022',
        help="Teacher model name (defaults to provider-appropriate model: Anthropic→claude-3-5-haiku-20241022, OpenAI→gpt-4-turbo-preview)"
    )
    parser.add_argument(
        '--allow_no_teacher',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='If true, allow running without a teacher API key (falls back to a simple heuristic scorer).',
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
    parser.add_argument(
        '--generation_temperature',
        type=float,
        default=0.2,
        help='Generation temperature for both models. Use 0 for greedy/deterministic generation.',
    )
    parser.add_argument(
        '--generation_top_p',
        type=float,
        default=0.95,
        help='Generation top_p for both models (when temperature > 0).',
    )
    parser.add_argument(
        '--generation_top_k',
        type=int,
        default=50,
        help='Generation top_k for both models (when temperature > 0).',
    )
    parser.add_argument(
        '--generation_seed',
        type=int,
        default=0,
        help='Random seed for reproducible generation. Use -1 to disable seeding.',
    )
    parser.add_argument(
        '--num_generations_per_prompt',
        type=int,
        default=1,
        help='Number of generations per prompt per model (default: 1).',
    )
    parser.add_argument(
        '--aggregate',
        type=str,
        default='mean',
        choices=['mean', 'max'],
        help="How to aggregate multiple generations into a single score per prompt (default: mean).",
    )
    parser.add_argument(
        '--score_mode',
        type=str,
        default='absolute',
        choices=['absolute', 'normalized', 'both'],
        help="Scoring mode: 'absolute' uses teacher score on student code; "
             "'normalized' uses student_score/teacher_score; "
             "'both' records both.",
    )
    parser.add_argument(
        '--dedupe_prompts',
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, de-duplicate prompts by (prompt, language) to avoid double-counting repeats in eval sets.",
    )
    parser.add_argument(
        '--tensorboard_dir',
        type=str,
        default=None,
        help="Optional TensorBoard log dir for validation charts (e.g., ./logs/tensorboard/validation).",
    )
    
    args = parser.parse_args()
    
    # Load test prompts
    test_prompts = load_test_prompts(args.test_prompts)
    if bool(args.dedupe_prompts):
        test_prompts = dedupe_test_prompts(test_prompts)
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
        generation_temperature=float(args.generation_temperature),
        generation_top_p=float(args.generation_top_p),
        generation_top_k=int(args.generation_top_k),
        generation_seed=(None if int(args.generation_seed) < 0 else int(args.generation_seed)),
        allow_no_teacher=bool(args.allow_no_teacher),
        score_mode=str(args.score_mode),
        tensorboard_dir=args.tensorboard_dir,
    )
    
    # Run validation
    results = validator.validate(
        test_prompts,
        num_generations_per_prompt=int(args.num_generations_per_prompt),
        aggregate=str(args.aggregate),
    )
    
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
        # Record the resolved MLX path actually used (default can be inferred from fine_tuned_path).
        'fine_tuned_mlx_path': (validator.fine_tuned_mlx_path if bool(args.use_mlx) else None),
        'test_prompts_file': args.test_prompts,
        'num_test_cases': len(test_prompts),
        'validation_date': datetime.now().isoformat(),
        'teacher_provider': args.teacher_provider,
        'teacher_model': args.teacher_model,
        'generation': {
            'temperature': float(args.generation_temperature),
            'top_p': float(args.generation_top_p),
            'top_k': int(args.generation_top_k),
            'seed': (None if int(args.generation_seed) < 0 else int(args.generation_seed)),
            'num_generations_per_prompt': int(args.num_generations_per_prompt),
            'aggregate': str(args.aggregate),
        },
        'score_mode': str(args.score_mode),
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nValidation results saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()

