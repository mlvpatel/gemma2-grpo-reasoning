# Gemma2 GRPO Reasoning

Fine-tuning Gemma 2 2B-IT for mathematical reasoning using Group Relative Policy Optimization (GRPO) with Tunix on TPU.

## Overview

This repository implements GRPO-based reinforcement learning to train Gemma 2 to produce structured, step-by-step reasoning for mathematical problems. The model learns to generate responses in a specific XML format with explicit reasoning chains.

### Architecture

- **Hardware**: TPU v5e-8 (2x4 mesh configuration)
- **Base Model**: Gemma 2 2B-IT
- **Fine-Tuning**: LoRA (rank=64) on Attention and MLP layers
- **Algorithm**: GRPO with 16 parallel generations per prompt
- **Dataset**: GSM8K (Grade School Math)

### Reward Structure

The training uses a weighted multi-component reward system:

| Component | Weight | Description |
|-----------|--------|-------------|
| Format | 25% | Correct XML tag structure (`<reasoning>`, `<answer>`) |
| Logic | 30% | Step-by-step reasoning with transition words |
| Accuracy | 45% | Correct numerical answer |
| Self-Correction | Bonus | Detecting and fixing errors mid-reasoning |
| Length | Bonus | Appropriate response length |

## Repository Structure

```
gemma2-grpo-reasoning/
├── notebooks/
│   └── train_grpo.ipynb      # Main training notebook
├── src/
│   ├── config.py             # Configuration parameters
│   ├── rewards.py            # Reward function implementations
│   ├── data.py               # Data loading utilities
│   └── utils.py              # Helper functions
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

- Python 3.10+
- TPU v5e-8 (recommended) or equivalent
- JAX with TPU support
- Tunix
- Qwix (for LoRA)

## Installation

```bash
pip install "jax[tpu]>=0.8.0" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install git+https://github.com/google/tunix
pip install git+https://github.com/google/qwix
pip install git+https://github.com/google/flax
pip install transformers datasets grain kagglehub
```

## Usage

### Training on Kaggle

1. Create a new Kaggle notebook with TPU v5e-8 accelerator
2. Upload `notebooks/train_grpo.ipynb`
3. Add Kaggle secret `HF_TOKEN` with your Hugging Face token
4. Run all cells (first run requires kernel restart after package installation)

### Configuration

Key parameters in `src/config.py`:

```python
# GRPO Settings
NUM_GENERATIONS = 16    # Parallel generations per prompt
NUM_ITERATIONS = 3      # Iterations per batch
BETA = 0.04             # KL penalty coefficient
EPSILON = 0.2           # Clipping parameter

# Training
MAX_STEPS = 600
LEARNING_RATE = 2e-6
LORA_RANK = 64

# Data
NUM_TRAIN_SAMPLES = 448
NUM_TEST_SAMPLES = 64
```

### Expected Output Format

The trained model produces responses in this structure:

```xml
<reasoning>
Step 1: Identify what we need to find.
Step 2: Set up the calculation.
Step 3: 150 × 4 = 600
Therefore, the total revenue is 600.
</reasoning>
<answer>
600
</answer>
```

## Technical Details

### Mesh Configuration

The (2,4) mesh configuration is optimized for Gemma 2's architecture:
- TP=4 splits the 4 KV-heads (1 head per core)
- FSDP=2 splits data and weights across 2 groups

### Memory Management

- CPU offloading during model initialization prevents TPU OOM
- bf16 precision reduces memory footprint
- Micro-batching (size=1) enables gradient accumulation

### GRPO Algorithm

GRPO generates multiple responses per prompt and computes relative advantages within each group, eliminating the need for a separate critic model. This approach:
- Reduces memory usage compared to PPO
- Provides stable training signal through group normalization
- Scales efficiently with parallel generation

## Results

Training on GSM8K with the default configuration typically shows:
- Format compliance improvement within first 100 steps
- Reasoning quality improvement by step 300
- Answer accuracy gains through step 600

## License

Apache License 2.0

## References

- [GRPO Paper](https://arxiv.org/abs/2402.03300) - DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- [Tunix](https://github.com/google/tunix) - JAX-native LLM post-training library
- [Gemma 2](https://ai.google.dev/gemma) - Google's open language model
- [GSM8K](https://github.com/openai/grade-school-math) - Grade school math dataset

## Author

Malav Patel ([@mlvpatel](https://github.com/mlvpatel))
