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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelValidator:
    """Validate and compare baseline vs fine-tuned models"""
    
    def __init__(self, base_model: str, fine_tuned_path: str, teacher_provider: str = "openai", teacher_model: str = "gpt-4-turbo-preview"):
        self.base_model_name = base_model
        self.fine_tuned_path = fine_tuned_path
        self.teacher_provider = teacher_provider
        self.teacher_model = teacher_model
        
        # Load tokenizers
        logger.info(f"Loading tokenizer from {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load baseline model
        logger.info(f"Loading baseline model: {base_model}")
        self.baseline_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            dtype=torch.bfloat16,  # Use dtype instead of torch_dtype
        )
        
        # Load fine-tuned model
        logger.info(f"Loading fine-tuned model from {fine_tuned_path}")
        self.fine_tuned_model = AutoModelForCausalLM.from_pretrained(
            fine_tuned_path,
            device_map="auto",
            dtype=torch.bfloat16,  # Use dtype instead of torch_dtype
        )
        
        # Initialize teacher for scoring
        self.teacher = self._init_teacher()
    
    def _init_teacher(self):
        """Initialize teacher model for scoring"""
        try:
            from train_rfai import TeacherModel
            api_key_env = "OPENAI_API_KEY" if self.teacher_provider == "openai" else "ANTHROPIC_API_KEY"
            return TeacherModel(self.teacher_provider, self.teacher_model, api_key_env)
        except Exception as e:
            logger.warning(f"Could not initialize teacher model: {e}")
            return None
    
    def generate_code(self, model, prompt: str, language: str, max_tokens: int = 512) -> str:
        """Generate code from a model"""
        formatted_prompt = f"Write high-quality {language} code:\n\n{prompt}\n\nCode:"
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
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
            baseline_code = self.generate_code(self.baseline_model, prompt, language)
            baseline_score = self.score_code(baseline_code, prompt, language)
            
            # Generate from fine-tuned
            fine_tuned_code = self.generate_code(self.fine_tuned_model, prompt, language)
            fine_tuned_score = self.score_code(fine_tuned_code, prompt, language)
            
            improvement = fine_tuned_score - baseline_score
            
            results['baseline_scores'].append(baseline_score)
            results['fine_tuned_scores'].append(fine_tuned_score)
            results['improvements'].append(improvement)
            
            # Store example
            results['examples'].append({
                'prompt': prompt,
                'language': language,
                'baseline_code': baseline_code[:500],  # Truncate for display
                'fine_tuned_code': fine_tuned_code[:500],
                'baseline_score': baseline_score,
                'fine_tuned_score': fine_tuned_score,
                'improvement': improvement
            })
            
            logger.info(f"  Baseline Score: {baseline_score:.4f}")
            logger.info(f"  Fine-tuned Score: {fine_tuned_score:.4f}")
            logger.info(f"  Improvement: {improvement:+.4f}")
        
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
        print("VALIDATION REPORT: Pre-Training vs Post-Training")
        print("="*80)
        print(f"\nTest Cases: {len(results['baseline_scores'])}")
        print(f"\nAverage Scores:")
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
            print(f"\nBaseline Code:")
            print("-" * 40)
            print(example['baseline_code'][:300] + "..." if len(example['baseline_code']) > 300 else example['baseline_code'])
            print(f"\nFine-tuned Code:")
            print("-" * 40)
            print(example['fine_tuned_code'][:300] + "..." if len(example['fine_tuned_code']) > 300 else example['fine_tuned_code'])
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
        default='Qwen/Qwen2.5-7B-Instruct',
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
        default='gpt-4-turbo-preview',
        help='Teacher model name'
    )
    
    args = parser.parse_args()
    
    # Load test prompts
    test_prompts = load_test_prompts(args.test_prompts)
    logger.info(f"Loaded {len(test_prompts)} test prompts")
    
    # Initialize validator
    validator = ModelValidator(
        base_model=args.base_model,
        fine_tuned_path=args.fine_tuned_path,
        teacher_provider=args.teacher_provider,
        teacher_model=args.teacher_model
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

