# Teacher Token Optimization

## Problem

1.19M input tokens for 350 samples is very high (~3,400 tokens per sample). This is expensive and slow.

## Root Causes

1. **Full rubric sent per request**: The entire scoring rubric (100+ lines) is included in every API call
2. **Full prompt text**: Complete prompt is sent even though only minimal context is needed for scoring
3. **No filtering**: All samples sent to teacher, even obviously broken ones
4. **Inefficient caching**: Cache keys use full strings instead of hashes

## Optimizations Implemented

### 1. Hash-Based Caching ✅
- **Before**: Cache keys used full prompt + code strings (memory intensive)
- **After**: Uses MD5 hashes (12 chars each) for prompt and code
- **Impact**: Reduces cache memory by ~90% while maintaining correctness
- **Implementation**: `STUDENT:{prompt_hash}:{code_hash}:{language}`

### 2. Prompt Truncation ✅
- **Before**: Full prompt sent to teacher (~500-1000 tokens)
- **After**: Only first 200 characters of prompt (minimal context)
- **Impact**: Reduces prompt tokens by ~70-80%
- **Config**: `truncate_prompt_for_scoring: true`, `prompt_context_chars: 200`

### 3. Rubric Moved to System Prompt ✅
- **Before**: Full rubric (100+ lines) included in every user message
- **After**: Rubric moved to system prompt (sent once per conversation)
- **Impact**: Reduces user message tokens by ~80-90%
- **Config**: `move_rubric_to_system_prompt: true`
- **Note**: System prompt is sent once per API session, not per-request

### 4. Tiered Scoring with Heuristic Filter ✅
- **Before**: All samples sent to teacher API
- **After**: Quick heuristic filter → only finalists sent to teacher
- **Heuristic checks**:
  - Code length (rejects <10 chars)
  - Balanced braces/parentheses (syntax check)
  - Basic structure (functions, classes)
  - Documentation presence
- **Impact**: Filters out ~30-50% of low-quality samples before API call
- **Config**: `use_tiered_scoring: true`, `heuristic_score_threshold: 0.3`

## Expected Token Reduction

### Per Sample (Before → After):
- **Rubric**: ~800 tokens → 0 tokens (moved to system prompt)
- **Prompt**: ~500 tokens → ~50 tokens (truncated to 200 chars)
- **Code**: ~500 tokens → ~500 tokens (unchanged)
- **Total**: ~1,800 tokens → ~550 tokens (**~70% reduction**)

### With Tiered Scoring:
- **Heuristic filter**: Rejects ~40% of samples
- **Effective reduction**: ~70% token reduction on remaining 60% = **~82% total reduction**

### Projected Results:
- **Before**: 1.19M tokens for 350 samples = 3,400 tokens/sample
- **After**: ~210K tokens for 350 samples = 600 tokens/sample
- **Savings**: ~980K tokens (**82% reduction**)

## Configuration

All optimizations are enabled by default in `config.yaml`:

```yaml
rlaif:
  # Teacher token optimization
  use_tiered_scoring: true # Use heuristic filter before teacher scoring
  heuristic_score_threshold: 0.3 # Only send samples above this to teacher
  truncate_prompt_for_scoring: true # Truncate prompt to minimal context
  prompt_context_chars: 200 # Max characters of prompt context
  move_rubric_to_system_prompt: true # Move rubric to system prompt
```

## Implementation Details

### Hash-Based Caching
```python
# Before
student_code_key = f"{sample['code']}:{sample['prompt']}:{sample['language']}"

# After
prompt_hash = hashlib.md5(sample['prompt'].encode()).hexdigest()[:12]
code_hash = hashlib.md5(sample['code'].encode()).hexdigest()[:12]
student_code_key = f"STUDENT:{prompt_hash}:{code_hash}:{sample['language']}"
```

### Prompt Truncation
```python
def _truncate_prompt(self, prompt: str, max_chars: int = 200) -> str:
    """Truncate prompt to minimal context needed for scoring"""
    if len(prompt) <= max_chars:
        return prompt
    truncated = prompt[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.7:
        truncated = truncated[:last_space]
    return truncated + "..."
```

### Tiered Scoring
```python
def _heuristic_score(self, code: str, language: str) -> float:
    """Quick heuristic filter to avoid sending low-quality code to teacher"""
    # Checks: length, syntax, structure, documentation
    # Returns: 0.0-1.0 score
    # If score < threshold, skip teacher API call
```

### Rubric in System Prompt
```python
# System prompt (sent once, not per-request)
score_system_prompt = f"""You are a strict scoring function for {language} code.
SCORING RUBRIC:
1. Correctness (0.3): ...
2. Code Quality (0.3): ...
3. Efficiency (0.2): ...
4. Documentation (0.2): ...
[Full rubric details...]
"""

# User prompt (minimal, per-request)
scoring_prompt = f"""Code:
```{language}
{code}
```

Context: {truncated_prompt}

Score:"""
```

## Benefits

1. **Cost Reduction**: ~82% fewer tokens = ~82% lower API costs
2. **Speed Improvement**: Fewer tokens = faster API responses
3. **Quality Maintained**: Heuristic filter only rejects obviously broken code
4. **Memory Efficiency**: Hash-based caching uses ~90% less memory
5. **Scalability**: Can process more samples with same token budget

## Monitoring

Track these metrics to verify optimization effectiveness:
- `api_tokens_sent`: Should drop significantly
- `teacher_score_calls`: Should drop with tiered scoring
- `heuristic_filter_rejects`: New metric for filtered samples
- Cache hit rate: Should remain high with hash-based caching






