# Model Validation and Hugging Face Upload Guide

## Overview

This guide explains how to:
1. Validate your fine-tuned model against the baseline
2. Upload your model to Hugging Face MLX Community
3. Compare pre-training vs post-training quality

## Model Validation

### Quick Start

```bash
python validate_model.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --fine_tuned_path ./checkpoints/checkpoint-500 \
    --test_prompts ./data/eval.jsonl
```

### What It Does

The validation script:
1. Loads both baseline and fine-tuned models
2. Generates code for each test prompt from both models
3. Scores outputs using the teacher model (OpenAI/Claude)
4. Calculates improvement statistics
5. Displays example comparisons

### Output

The script generates:
- **Console Report**: Summary statistics and top examples
- **JSON Results**: Detailed results saved to `validation_results.json`

### Example Output

```
================================================================================
VALIDATION REPORT: Pre-Training vs Post-Training
================================================================================

Test Cases: 5

Average Scores:
  Baseline Model:    0.6234
  Fine-tuned Model:  0.7891
  Average Improvement: +0.1657 (+26.58%)

Improvement Distribution:
  Improved:  4 cases
  Degraded:  1 cases
  Equal:     0 cases
```

### Custom Test Prompts

Create a JSONL file with your test cases:

```json
{"prompt": "Implement a binary search function", "language": "python"}
{"prompt": "Create a thread-safe queue", "language": "cpp"}
{"prompt": "Write a JSON parser", "language": "rust"}
```

Then run:
```bash
python validate_model.py \
    --fine_tuned_path ./checkpoints/checkpoint-500 \
    --test_prompts ./my_test_prompts.jsonl
```

## Hugging Face Upload

### Prerequisites

1. **Hugging Face Account**: Sign up at https://huggingface.co
2. **Access Token**: Get your token from https://huggingface.co/settings/tokens
3. **MLX Community Access**: Request to join https://huggingface.co/mlx-community

### Setup

1. Set your Hugging Face token:
   ```bash
   export HUGGINGFACE_TOKEN="hf_..."
   ```

2. Configure upload in `config.yaml`:
   ```yaml
   huggingface:
     upload_to_hub: true
     repo_id: "mlx-community/qwen-code-rfai"  # Your model name
     hf_token_env: "HUGGINGFACE_TOKEN"
     upload_quantized: true
     private: false
   ```

### Automatic Upload

Models are automatically uploaded:
- After training completes (final checkpoint)
- At specified intervals during training

### Manual Upload

You can also upload manually using `mlx-lm`:

```bash
mlx_lm.convert \
    --hf-path ./checkpoints/checkpoint-500 \
    --upload-repo mlx-community/qwen-code-rfai \
    -q  # For quantization
```

### Model Card

The uploaded model includes an automatic model card with:
- Training details
- Base model information
- Training statistics
- Usage examples
- Model tags

### Repository Structure

Your uploaded model will have:
```
mlx-community/qwen-code-rfai/
├── README.md              # Model card
├── config.json            # Model configuration
├── model.safetensors      # MLX weights
├── tokenizer files        # Tokenizer configuration
└── training_stats.json    # Training statistics
```

## Best Practices

### Validation

1. **Use Diverse Test Cases**: Include examples from all languages (Python, C++, Rust)
2. **Test Edge Cases**: Include complex prompts and simple ones
3. **Compare Multiple Checkpoints**: Validate at different training stages
4. **Save Results**: Keep validation results for tracking progress

### Hugging Face Upload

1. **Choose Good Names**: Use descriptive repo IDs like `qwen-code-rfai-v1`
2. **Add Tags**: Update model card tags for discoverability
3. **Include Examples**: Add usage examples in the model card
4. **Version Control**: Use tags for different model versions

## Troubleshooting

### Validation Issues

**"Teacher model not available"**
- Set API keys: `export OPENAI_API_KEY="..."` or `export ANTHROPIC_API_KEY="..."`
- The script will use fallback scoring if teacher is unavailable

**"Out of memory"**
- Use smaller test sets
- Reduce `max_tokens` in generation

### Upload Issues

**"Authentication failed"**
- Check your `HUGGINGFACE_TOKEN` is set correctly
- Verify token has write permissions

**"Repository not found"**
- Ensure you've joined the MLX Community organization
- Check repository name is correct (format: `mlx-community/model-name`)

**"Upload timeout"**
- Large models may take time to upload
- Check your internet connection
- Try uploading during off-peak hours

## Example Workflow

1. **Train Model**:
   ```bash
   python train_rfai.py --config config.yaml
   ```

2. **Validate Model**:
   ```bash
   python validate_model.py \
       --fine_tuned_path ./checkpoints/checkpoint-500 \
       --output ./validation_results.json
   ```

3. **Review Results**:
   - Check console output for summary
   - Review `validation_results.json` for details
   - Compare examples to see improvements

4. **Upload to HF** (if configured):
   - Model uploads automatically after training
   - Or manually upload using mlx-lm convert

5. **Share Model**:
   - Share the Hugging Face model link
   - Others can use: `mlx_lm.load("mlx-community/qwen-code-rfai")`

## Dataset Collection

During training, datasets are automatically collected and can be uploaded to Hugging Face. See [DATASET_GUIDE.md](./DATASET_GUIDE.md) for details.

## References

- [MLX Community](https://huggingface.co/mlx-community)
- [MLX-LM Documentation](https://github.com/ml-explore/mlx-examples)
- [Hugging Face Model Cards](https://huggingface.co/docs/hub/model-cards)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

