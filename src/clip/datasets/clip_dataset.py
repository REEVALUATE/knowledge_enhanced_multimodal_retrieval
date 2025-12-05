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


class CLIPTrainDataset(Dataset):
    """
    Training dataset for CLIP with query-target pairs.
    
    Returns: (image, query, target_text, uuid)
    """
    
    def __init__(
        self,
        uuids: List[str],
        image_folder: str,
        text_folder: str,
        preprocessor=None,
        max_text_length: int = 150
    ):
        """
        Args:
            uuids: List of sample UUIDs
            image_folder: Path to images
            text_folder: Path to text JSON files (new format with query/target_text)
            preprocessor: Image preprocessing function
            max_text_length: Maximum number of words in text
        """
        self.uuids = uuids
        self.image_folder = Path(image_folder)
        self.text_folder = Path(text_folder)
        self.preprocessor = preprocessor
        self.max_text_length = max_text_length
        
        logger.info(f"Training dataset initialized: {len(uuids)} samples")
    
    def __len__(self):
        return len(self.uuids)
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_text_length words."""
        words = text.split()
        if len(words) > self.max_text_length:
            return " ".join(words[:self.max_text_length])
        return text
    
    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        
        # Load image
        image_path = self.image_folder / f"{uuid}.jpg"
        if not image_path.exists():
            for ext in ['.jpeg', '.png']:
                alt_path = self.image_folder / f"{uuid}{ext}"
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.preprocessor:
                image = self.preprocessor(image)
        except Exception as e:
            logger.error(f"Error loading image {uuid}: {e}")
            if self.preprocessor:
                image = torch.zeros(3, 224, 224)
            else:
                image = Image.new('RGB', (224, 224))
        
        # Load texts (new format)
        text_path = self.text_folder / f"{uuid}.json"
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                query = self._truncate_text(data.get('query', ''))
                target_text = self._truncate_text(data.get('target_text', ''))
        except Exception as e:
            logger.error(f"Error loading text for {uuid}: {e}")
            query = ""
            target_text = ""
        
        return image, query, target_text, uuid


class CLIPEvalDataset(Dataset):
    """
    Evaluation dataset for CLIP with query-target pairs.
    
    Returns: (image, query, target_text)
    """
    
    def __init__(
        self,
        uuids: List[str],
        image_folder: str,
        text_folder: str,
        preprocessor=None,
        max_text_length: int = 150
    ):
        self.uuids = uuids
        self.image_folder = Path(image_folder)
        self.text_folder = Path(text_folder)
        self.preprocessor = preprocessor
        self.max_text_length = max_text_length
        
        logger.info(f"Evaluation dataset initialized: {len(uuids)} samples")
    
    def __len__(self):
        return len(self.uuids)
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_text_length words."""
        words = text.split()
        if len(words) > self.max_text_length:
            return " ".join(words[:self.max_text_length])
        return text
    
    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        
        # Load image
        image_path = self.image_folder / f"{uuid}.jpg"
        if not image_path.exists():
            for ext in ['.jpeg', '.png']:
                alt_path = self.image_folder / f"{uuid}{ext}"
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.preprocessor:
                image = self.preprocessor(image) #uncommented for all other clip model
                # HuggingFace CLIPProcessor requires explicit keyword
                # processed = self.preprocessor(images=image, return_tensors="pt")
                # image = processed["pixel_values"].squeeze(0)   # (3,224,224)
        except Exception as e:
            logger.error(f"Error loading image {uuid}: {e}")
            if self.preprocessor:
                image = torch.zeros(3, 224, 224)
            else:
                image = Image.new('RGB', (224, 224))
        
        # Load texts
        text_path = self.text_folder / f"{uuid}.json"
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                query = self._truncate_text(data.get('query', ''))
                target_text = self._truncate_text(data.get('target_text', ''))
                uuid = data.get('uuid', uuid)
        except Exception as e:
            logger.error(f"Error loading text for {uuid}: {e}")
            query = ""
            target_text = ""
        
        return image, query, target_text, uuid

class TextOnlyDataset(Dataset):
    """
    Dataset for text-only models (MPNet, E5, GTE).
    Only loads query and target_text, no images.
    """
    
    def __init__(
        self,
        uuids: List[str],
        text_folder: str,
        max_text_length: int = 150
    ):
        self.uuids = uuids
        self.text_folder = Path(text_folder)
        self.max_text_length = max_text_length
        
        logger.info(f"Text-only dataset initialized: {len(uuids)} samples")
    
    def __len__(self):
        return len(self.uuids)
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_text_length words."""
        words = text.split()
        if len(words) > self.max_text_length:
            return " ".join(words[:self.max_text_length])
        return text
    
    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        
        text_path = self.text_folder / f"{uuid}.json"
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                query = self._truncate_text(data.get('query', ''))
                target_text = self._truncate_text(data.get('target_text', ''))
        except Exception as e:
            logger.error(f"Error loading text for {uuid}: {e}")
            query = ""
            target_text = ""
        
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