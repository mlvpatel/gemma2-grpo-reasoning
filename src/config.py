"""Configuration parameters for GRPO training."""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """Training configuration for Gemma2 GRPO."""
    
    # Model
    model_version: str = "gemma2-2b-it"
    model_path: str = "google/gemma-2/flax/gemma2-2b-it"
    model_hf_name: str = "google/gemma-2-2b-it"
    
    # Mesh configuration for TPU v5e-8
    # tp=4 splits 4 KV-heads, fsdp=2 splits data
    mesh_shape: Tuple[int, int] = (2, 4)
    mesh_axes: Tuple[str, str] = ("fsdp", "tp")
    
    # LoRA configuration
    lora_rank: int = 64
    lora_alpha: float = 64.0
    lora_target_pattern: str = ".*q_einsum|.*kv_einsum|.*o_proj|.*gate_proj|.*up_proj|.*down_proj"
    
    # GRPO parameters
    num_generations: int = 16  # G in paper
    num_iterations: int = 3    # mu in paper
    beta: float = 0.04         # KL penalty
    epsilon: float = 0.2       # Clipping
    
    # Training
    max_steps: int = 600
    learning_rate: float = 2e-6
    warmup_steps: int = 40
    weight_decay: float = 0.01
    mini_batch_size: int = 8
    micro_batch_size: int = 1
    
    # Sequence lengths
    max_prompt_length: int = 256
    max_generation_length: int = 768
    
    # Reward weights
    reward_weight_format: float = 0.25
    reward_weight_logic: float = 0.30
    reward_weight_accuracy: float = 0.45
    
    # Output tags
    tag_reasoning_start: str = "<reasoning>"
    tag_reasoning_end: str = "</reasoning>"
    tag_answer_start: str = "<answer>"
    tag_answer_end: str = "</answer>"
    
    # Paths
    checkpoint_dir: str = "/kaggle/working/checkpoints"
    output_dir: str = "/kaggle/working/gemma2-grpo-output"
    
    # Data
    num_train_samples: int = 448
    num_test_samples: int = 64
    random_seed: int = 42
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    @property
    def system_prompt(self) -> str:
        """Generate the system prompt for training."""
        return f"""You are a mathematical problem solver who shows their work.

For each problem:
1. Think through it step-by-step inside {self.tag_reasoning_start} and {self.tag_reasoning_end} tags
2. Show your calculations clearly
3. Give your final numerical answer inside {self.tag_answer_start} and {self.tag_answer_end} tags

Example:
{self.tag_reasoning_start}
Step 1: Identify what we need to find.
Step 2: Set up the calculation.
Step 3: 5 × 10 = 50
Therefore, the answer is 50.
{self.tag_reasoning_end}
{self.tag_answer_start}
50
{self.tag_answer_end}"""
    
    def format_prompt(self, question: str) -> str:
        """Format a question with the system prompt."""
        return f"{self.system_prompt}\n\nQuestion: {question}\nSolution:"


# Default configuration instance
default_config = Config()
