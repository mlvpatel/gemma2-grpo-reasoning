"""Data loading utilities for GSM8K dataset."""

from typing import List, Dict, Any, Optional
from datasets import load_dataset
import grain

from .config import Config


class GSM8KDataSource:
    """Data source compatible with Grain MapDataset."""
    
    def __init__(self, data: List[Dict[str, str]]):
        """Initialize data source.
        
        Args:
            data: List of dicts with 'prompt' and 'answer' keys
        """
        self._data = data
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """Get item by index.
        
        Returns dict with 'prompts' and 'answers' keys
        (plural form required by Tunix).
        """
        item = self._data[idx]
        return {
            'prompts': item['prompt'],
            'answers': item['answer']
        }


class GSM8KDataLoader:
    """Data loader for GSM8K dataset."""
    
    def __init__(self, config: Config = None):
        """Initialize data loader.
        
        Args:
            config: Configuration object. Uses default if None.
        """
        self.config = config or Config()
        self._train_data: Optional[List[Dict]] = None
        self._test_data: Optional[List[Dict]] = None
    
    @staticmethod
    def extract_answer(answer_text: str) -> str:
        """Extract final answer from GSM8K format.
        
        GSM8K answers have format: "reasoning text #### final_answer"
        
        Args:
            answer_text: Raw answer string from dataset
            
        Returns:
            Extracted final answer
        """
        if '####' in answer_text:
            return answer_text.split('####')[-1].strip()
        return answer_text.strip()
    
    def load(self) -> tuple:
        """Load and prepare GSM8K dataset.
        
        Returns:
            Tuple of (train_data, test_data) lists
        """
        cfg = self.config
        
        # Load from HuggingFace
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        dataset = dataset.shuffle(seed=cfg.random_seed)
        
        # Prepare data
        all_data = []
        total_needed = cfg.num_train_samples + cfg.num_test_samples
        
        for i, item in enumerate(dataset):
            if i >= total_needed:
                break
            all_data.append({
                'prompt': cfg.format_prompt(item['question']),
                'answer': self.extract_answer(item['answer'])
            })
        
        self._train_data = all_data[:cfg.num_train_samples]
        self._test_data = all_data[cfg.num_train_samples:total_needed]
        
        return self._train_data, self._test_data
    
    def get_grain_datasets(self) -> tuple:
        """Get Grain MapDataset objects for training.
        
        Returns:
            Tuple of (train_dataset, val_dataset) Grain MapDatasets
        """
        if self._train_data is None:
            self.load()
        
        cfg = self.config
        
        train_dataset = (
            grain.MapDataset.source(GSM8KDataSource(self._train_data))
            .shuffle(seed=cfg.random_seed)
        )
        
        val_dataset = (
            grain.MapDataset.source(GSM8KDataSource(self._test_data))
        )
        
        return train_dataset, val_dataset
    
    @property
    def train_data(self) -> List[Dict]:
        """Get raw training data."""
        if self._train_data is None:
            self.load()
        return self._train_data
    
    @property
    def test_data(self) -> List[Dict]:
        """Get raw test data."""
        if self._test_data is None:
            self.load()
        return self._test_data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dict with dataset statistics
        """
        if self._train_data is None:
            self.load()
        
        cfg = self.config
        
        return {
            'num_train': len(self._train_data),
            'num_test': len(self._test_data),
            'full_batch_train': len(self._train_data) * cfg.num_generations,
            'full_batch_test': len(self._test_data) * cfg.num_generations,
            'avg_prompt_length': sum(len(d['prompt']) for d in self._train_data) / len(self._train_data),
        }
