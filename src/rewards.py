"""Reward functions for GRPO training.

Implements a weighted multi-component reward system:
- Format (25%): Correct XML tag structure
- Logic (30%): Step-by-step reasoning quality
- Accuracy (45%): Correct numerical answer
- Self-correction (bonus): Error detection and correction
- Length (bonus): Appropriate response length
"""

import re
from typing import List, Any
from .config import Config


class RewardFunctions:
    """Collection of reward functions for GRPO training."""
    
    def __init__(self, config: Config = None):
        """Initialize reward functions with configuration.
        
        Args:
            config: Configuration object. Uses default if None.
        """
        self.config = config or Config()
    
    def format_reward(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """Reward for correct XML tag structure.
        
        Checks for proper <reasoning>...</reasoning> followed by <answer>...</answer>
        
        Args:
            prompts: List of input prompts (unused but required by interface)
            completions: List of model completions to evaluate
            
        Returns:
            List of weighted reward scores (max = 0.25)
        """
        cfg = self.config
        rewards = []
        
        for c in completions:
            score = 0.0
            has_r_start = cfg.tag_reasoning_start in c
            has_r_end = cfg.tag_reasoning_end in c
            has_a_start = cfg.tag_answer_start in c
            has_a_end = cfg.tag_answer_end in c
            
            if has_r_start and has_r_end and has_a_start and has_a_end:
                r_end_pos = c.find(cfg.tag_reasoning_end)
                a_start_pos = c.find(cfg.tag_answer_start)
                if r_end_pos < a_start_pos:
                    score = 1.0
                else:
                    score = 0.5
            elif (has_r_start and has_r_end) or (has_a_start and has_a_end):
                score = 0.3
            
            rewards.append(score * cfg.reward_weight_format)
        
        return rewards
    
    def logic_reward(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """Reward for quality reasoning with logical transitions.
        
        Evaluates:
        - Transition words (therefore, because, etc.)
        - Explicit steps (Step 1, Step 2, etc.)
        - Mathematical expressions
        - Equality statements
        
        Args:
            prompts: List of input prompts
            completions: List of model completions
            
        Returns:
            List of weighted reward scores (max = 0.30)
        """
        cfg = self.config
        TRANSITIONS = [
            'therefore', 'because', 'since', 'so', 'thus', 'hence',
            'first', 'second', 'third', 'then', 'next', 'finally',
            'step', 'calculate', 'compute', 'multiply', 'divide', 'add', 'subtract'
        ]
        
        rewards = []
        pattern = f'{cfg.tag_reasoning_start}(.*?){cfg.tag_reasoning_end}'
        
        for c in completions:
            score = 0.0
            match = re.search(pattern, c, re.DOTALL | re.IGNORECASE)
            
            if match:
                content = match.group(1).lower()
                
                # Transition words (up to 0.4)
                trans_count = sum(1 for t in TRANSITIONS if t in content)
                score += min(0.4, trans_count * 0.05)
                
                # Explicit steps (up to 0.3)
                step_count = len(re.findall(r'step\s*\d', content))
                score += min(0.3, step_count * 0.1)
                
                # Mathematical expressions (up to 0.2)
                math_ops = len(re.findall(r'\d+\s*[+\-×÷*/]\s*\d+', content))
                score += min(0.2, math_ops * 0.05)
                
                # Equality statements (up to 0.1)
                equals_count = content.count('=') + content.count('equals')
                score += min(0.1, equals_count * 0.02)
            
            rewards.append(min(1.0, score) * cfg.reward_weight_logic)
        
        return rewards
    
    def accuracy_reward(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """Reward for correct numerical answer.
        
        Compares extracted answer with ground truth, allowing for:
        - Exact match (1.0)
        - Within 1% (0.9)
        - Within 10% (0.5)
        
        Args:
            prompts: List of input prompts
            completions: List of model completions
            answers: List of ground truth answers (from kwargs)
            
        Returns:
            List of weighted reward scores (max = 0.45)
        """
        cfg = self.config
        answers = kwargs.get('answers', [])
        rewards = []
        pattern = f'{cfg.tag_answer_start}(.*?){cfg.tag_answer_end}'
        
        for i, c in enumerate(completions):
            score = 0.0
            
            if i >= len(answers):
                rewards.append(0.0)
                continue
            
            match = re.search(pattern, c, re.DOTALL)
            if not match:
                rewards.append(0.0)
                continue
            
            pred_text = match.group(1).strip()
            truth_text = str(answers[i]).strip()
            
            pred_nums = re.findall(r'-?\d+\.?\d*', pred_text)
            truth_nums = re.findall(r'-?\d+\.?\d*', truth_text)
            
            if pred_nums and truth_nums:
                try:
                    pred = float(pred_nums[-1])
                    truth = float(truth_nums[-1])
                    
                    if pred == truth:
                        score = 1.0
                    elif abs(truth) > 0.001:
                        ratio = pred / truth
                        if 0.99 <= ratio <= 1.01:
                            score = 0.9
                        elif 0.9 <= ratio <= 1.1:
                            score = 0.5
                except (ValueError, ZeroDivisionError):
                    pass
            
            rewards.append(score * cfg.reward_weight_accuracy)
        
        return rewards
    
    def self_correction_reward(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """Bonus reward for self-correction behavior.
        
        Detects phrases indicating the model caught and fixed errors.
        
        Args:
            prompts: List of input prompts
            completions: List of model completions
            
        Returns:
            List of bonus rewards (max = 0.10)
        """
        CORRECTION_PHRASES = [
            "wait", "actually", "let me recalculate", "i made an error",
            "correction", "re-checking", "that's not right", "let me redo"
        ]
        
        rewards = []
        for c in completions:
            c_lower = c.lower()
            count = sum(1 for phrase in CORRECTION_PHRASES if phrase in c_lower)
            rewards.append(min(0.1, count * 0.03))
        
        return rewards
    
    def length_reward(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """Bonus reward for appropriate response length.
        
        Ideal: 100-400 words
        Acceptable: 50-600 words
        
        Args:
            prompts: List of input prompts
            completions: List of model completions
            
        Returns:
            List of bonus rewards (max = 0.05)
        """
        rewards = []
        for c in completions:
            words = len(c.split())
            if 100 <= words <= 400:
                rewards.append(0.05)
            elif 50 <= words <= 600:
                rewards.append(0.02)
            else:
                rewards.append(0.0)
        
        return rewards
    
    def get_all_functions(self) -> List:
        """Get list of all reward functions for GRPO trainer.
        
        Returns:
            List of reward function callables
        """
        return [
            self.format_reward,
            self.logic_reward,
            self.accuracy_reward,
            self.self_correction_reward,
            self.length_reward,
        ]


def create_reward_functions(config: Config = None) -> List:
    """Factory function to create reward functions.
    
    Args:
        config: Configuration object
        
    Returns:
        List of reward function callables
    """
    rf = RewardFunctions(config)
    return rf.get_all_functions()
