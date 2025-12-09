"""
Dataset classes for CLIP fine-tuning with query-target pairs.
"""

import json
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

logger = logging.getLogger(__name__)


from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import torch

class CLIPTrainDatasetHF(Dataset):
    """
    Training dataset for CLIP using HuggingFace datasets.
    
    Returns: (image, query, target_text, uuid)
    """
    
    def __init__(
        self,
        hf_dataset,
        preprocessor=None,
        max_text_length: int = 150
    ):
        """
        Args:
            hf_dataset: HuggingFace dataset split (e.g., ds["train"])
            preprocessor: Image preprocessing function
            max_text_length: Maximum number of words in text
        """
        self.dataset = hf_dataset
        self.preprocessor = preprocessor
        self.max_text_length = max_text_length
        
        logger.info(f"Training dataset initialized: {len(self.dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_text_length words."""
        words = text.split()
        if len(words) > self.max_text_length:
            return " ".join(words[:self.max_text_length])
        return text
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Get image (already PIL Image from HF)
        try:
            image = sample['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.preprocessor:
                image = self.preprocessor(image)
        except Exception as e:
            logger.error(f"Error loading image {sample['uuid']}: {e}")
            if self.preprocessor:
                image = torch.zeros(3, 224, 224)
            else:
                image = Image.new('RGB', (224, 224))
        
        # Get texts
        query = self._truncate_text(sample['query_text'])
        target_text = self._truncate_text(sample['target_text'])
        uuid = sample['uuid']
        
        return image, query, target_text, uuid


class CLIPEvalDatasetHF(Dataset):
    """
    Evaluation dataset for CLIP using HuggingFace datasets.
    
    Returns: (image, query, target_text, uuid)
    """
    
    def __init__(
        self,
        hf_dataset,
        preprocessor=None,
        max_text_length: int = 150
    ):
        self.dataset = hf_dataset
        self.preprocessor = preprocessor
        self.max_text_length = max_text_length
        
        logger.info(f"Evaluation dataset initialized: {len(self.dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_text_length words."""
        words = text.split()
        if len(words) > self.max_text_length:
            return " ".join(words[:self.max_text_length])
        return text
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Get image
        try:
            image = sample['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.preprocessor:
                image = self.preprocessor(image)
        except Exception as e:
            logger.error(f"Error loading image {sample['uuid']}: {e}")
            if self.preprocessor:
                image = torch.zeros(3, 224, 224)
            else:
                image = Image.new('RGB', (224, 224))
        
        # Get texts
        query = self._truncate_text(sample['query_text'])
        target_text = self._truncate_text(sample['target_text'])
        uuid = sample['uuid']
        
        return image, query, target_text, uuid


class TextOnlyDatasetHF(Dataset):
    """
    Dataset for text-only models using HuggingFace datasets.
    Only loads query and target_text, no images.
    """
    
    def __init__(
        self,
        hf_dataset,
        max_text_length: int = 150
    ):
        self.dataset = hf_dataset
        self.max_text_length = max_text_length
        
        logger.info(f"Text-only dataset initialized: {len(self.dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_text_length words."""
        words = text.split()
        if len(words) > self.max_text_length:
            return " ".join(words[:self.max_text_length])
        return text
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        query = self._truncate_text(sample['query_text'])
        target_text = self._truncate_text(sample['target_text'])
        
        return query, target_text

def collate_fn_train(batch):
    """Collate function for training."""
    images, queries, targets, uuids = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(queries), list(targets), list(uuids)


def collate_fn_eval(batch):
    """Collate function for evaluation."""
    images, queries, targets, uuids = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(queries), list(targets), list(uuids)

def collate_fn_eval_texts(batch):
    """Collate function for evaluation."""
    queries, targets = zip(*batch)
    
    return list(queries), list(targets)