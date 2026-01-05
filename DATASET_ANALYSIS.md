# Dataset Size Analysis

## Current Dataset

- **Training prompts**: 100
- **Eval prompts**: 20
- **Samples per prompt**: 6
- **Effective training samples per epoch**: 600 (100 prompts × 6 samples)
- **Total unique prompts**: 100

## Is This Sufficient?

### ❌ **NO - Dataset is Too Small for Optimal Learning**

## Why More Data is Needed

### 1. **Limited Prompt Diversity**
- **Current**: 100 unique prompts
- **Problem**: Model sees the same prompts every epoch
- **Impact**: 
  - Overfitting to specific prompt patterns
  - Poor generalization to new problems
  - Limited exposure to different code patterns and idioms

### 2. **Training Evidence**
From your training analysis:
- **Reward regression** (E3: 0.2804 → E4: 0.2553) suggests overfitting
- **Best-of-N improving but average not** indicates model can generate good code but lacks consistency
- **Slow convergence** (only 2-5% improvement per epoch) suggests insufficient learning signal

### 3. **Industry Benchmarks**
For code generation fine-tuning:
- **Minimum viable**: 500-1000 unique prompts
- **Good results**: 1000-5000 unique prompts
- **Excellent results**: 5000+ unique prompts

### 4. **Your Current Setup**
- **100 prompts × 6 samples = 600 samples per epoch**
- **But only 100 unique patterns** to learn from
- **After 4 epochs**: Model has seen each prompt 4 times
- **Result**: Model memorizes patterns rather than learning generalizable skills

## Recommended Dataset Sizes

### **Minimum for Improvement** (Priority 1)
- **500-1000 unique prompts**
- **Expected impact**: 
  - Reward improvement: 0.25 → 0.35-0.45
  - Better generalization
  - Reduced overfitting

### **Good for Convergence** (Priority 2)
- **1000-2000 unique prompts**
- **Expected impact**:
  - Reward improvement: 0.25 → 0.50-0.60
  - Consistent improvement across epochs
  - Better code quality

### **Excellent for Production** (Priority 3)
- **2000-5000+ unique prompts**
- **Expected impact**:
  - Reward improvement: 0.25 → 0.65-0.75 (target)
  - Strong generalization
  - Production-ready quality

## Data Diversity Requirements

### Problem Types (Current: ~10 types)
**Recommended**: 20-30+ problem types
- Algorithms (sorting, searching, graph, DP)
- Data structures (trees, heaps, hash tables)
- System programming (memory, concurrency)
- Design patterns (singleton, factory, observer)
- API design (REST, async, error handling)
- Testing and validation
- Optimization and performance

### Difficulty Levels (Current: Mixed)
**Recommended**: Explicit difficulty distribution
- **Easy (30%)**: Simple functions, basic algorithms
- **Medium (50%)**: Moderate complexity, multiple concepts
- **Hard (20%)**: Complex systems, advanced patterns

### Language Distribution (Current: 3 languages)
**Recommended**: Maintain balanced distribution
- Python: 33-40%
- C++: 30-35%
- Rust: 25-30%

### Code Patterns (Current: Limited)
**Recommended**: Include diverse patterns
- Object-oriented vs functional
- Imperative vs declarative
- Error handling strategies
- Memory management approaches
- Concurrency patterns
- Testing methodologies

## How to Expand Your Dataset

### Option 1: Generate More Prompts (Recommended)
```python
# Use GPT-4/Claude to generate diverse prompts
# Focus on:
# - Different problem domains
# - Various difficulty levels
# - Multiple code patterns
# - Edge cases and special scenarios
```

### Option 2: Use Existing Code Datasets
- **CodeSearchNet**: 2M+ code snippets
- **The Stack**: Large code dataset
- **CodeXGLUE**: Code understanding tasks
- **HumanEval**: 164 Python problems (good for eval)

### Option 3: Augment Current Prompts
- **Paraphrase prompts**: Same problem, different wording
- **Add constraints**: "without using libraries", "with error handling"
- **Vary requirements**: Different input/output formats
- **Add context**: "for production use", "with unit tests"

### Option 4: Curriculum Learning Enhancement
- **Start easy**: Simple problems in early epochs
- **Increase difficulty**: Complex problems in later epochs
- **Mix difficulty**: Current approach, but with more examples

## Expected Impact of Dataset Expansion

### With 500 Prompts (5x increase)
- **Training samples per epoch**: 3000 (500 × 6)
- **Expected reward**: 0.35-0.45 (vs current 0.25)
- **Convergence**: More stable, less overfitting
- **Time per epoch**: ~2-3x longer (but better results)

### With 1000 Prompts (10x increase)
- **Training samples per epoch**: 6000 (1000 × 6)
- **Expected reward**: 0.50-0.60
- **Convergence**: Much more stable
- **Time per epoch**: ~4-5x longer

### With 2000 Prompts (20x increase)
- **Training samples per epoch**: 12000 (2000 × 6)
- **Expected reward**: 0.65-0.75 (target range)
- **Convergence**: Very stable
- **Time per epoch**: ~8-10x longer

## Trade-offs

### More Data = Better Results BUT:
- **Longer training time**: More prompts = more time per epoch
- **More API costs**: More teacher model calls for scoring
- **More storage**: Larger checkpoints and logs

### Recommendations:
1. **Start with 500 prompts** (5x current) - good balance
2. **Monitor training** - if still overfitting, increase to 1000
3. **Use data augmentation** - paraphrase existing prompts first
4. **Focus on diversity** - different problem types, not just more of the same

## Quick Wins (Low Effort, High Impact)

### 1. Paraphrase Existing Prompts (1-2 hours)
- Take your 100 prompts
- Use GPT-4/Claude to generate 2-3 variations each
- Result: 200-300 prompts with minimal effort

### 2. Add Constraints to Prompts (1 hour)
- "Implement X with error handling"
- "Implement X without using standard library Y"
- "Implement X with unit tests"
- Result: 2-3x more training signal from same prompts

### 3. Extract from Code Repositories (2-3 hours)
- Find GitHub repos with similar code patterns
- Extract function docstrings as prompts
- Filter for quality and diversity
- Result: 200-500 additional prompts

## Conclusion

**Your current dataset (100 prompts) is insufficient for optimal learning.**

### Immediate Action:
1. **Expand to at least 500 prompts** (5x current)
2. **Focus on diversity** over quantity
3. **Monitor training** - if reward still regresses, increase to 1000

### Expected Outcome:
- **With 500 prompts**: Reward should reach 0.35-0.45
- **With 1000 prompts**: Reward should reach 0.50-0.60
- **With 2000 prompts**: Reward should reach 0.65-0.75 (target)

The slow convergence and reward regression you're seeing are classic signs of insufficient training data. More diverse prompts will significantly improve learning and reward stability.









