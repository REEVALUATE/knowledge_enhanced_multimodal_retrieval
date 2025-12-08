"""
CLIP evaluation script with unified metrics computation.
Supports T2I, I2T, and T2T retrieval evaluation.
"""

import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip

import json
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

from ..model.clip_model import load_clip_model
from ..datasets.clip_dataset import collate_fn_eval
from ..utils.data_utils import get_data_splits, load_splits_from_json
from ..utils.logging_utils import setup_logger, save_metrics_to_json
from .metrics import compute_all_retrieval_metrics, compute_training_metrics

logger = logging.getLogger(__name__)

from huggingface_hub import login
login(token="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")
print("Logged in to Hugging Face Hub.")
def seed_worker(worker_id):
    """Worker seed function for DataLoader to ensure reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class CLIPEvaluationDataset(Dataset):
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
                # Check if HuggingFace processor or OpenAI CLIP preprocess
                if hasattr(self.preprocessor, 'image_processor'):
                    # HuggingFace CLIPProcessor
                    processed = self.preprocessor(images=image, return_tensors="pt")
                    image = processed["pixel_values"].squeeze(0)  # [3,224,224]
                else:
                    # OpenAI CLIP preprocess
                    image = self.preprocessor(image)
        except Exception as e:
            logger.error(f"Error loading image {uuid}: {e}")
            image = torch.zeros(3, 224, 224)
        
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
    

@torch.no_grad()
def evaluate_clip_model(
    model,
    preprocess,
    dataset: CLIPEvaluationDataset,
    batch_size: int = 64,
    device: str = 'cuda',
    seed: int = 42,
    tasks: List[str] = ["T2I", "I2T", "T2T"],
    compute_recall: bool = True,
    compute_mrr: bool = True
) -> Dict[str, float]:
    """
    Evaluate CLIP model on retrieval tasks.
    
    Args:
        model: CLIP model
        dataset: Evaluation dataset
        batch_size: Batch size
        device: Device ('cuda' or 'cpu')
        seed: Random seed for reproducibility
        tasks: List of tasks to evaluate (subset of ["T2I", "I2T", "T2T"])
        compute_recall: Whether to compute Recall@K metrics
        compute_mrr: Whether to compute MRR and Mean Rank
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Determine actual device
    use_cuda = torch.cuda.is_available()
    model_device = next(model.parameters()).device
    actual_device = str(model_device)
    
    if not use_cuda and device == 'cuda':
        logger.warning("CUDA requested but not available, using CPU")
    
    logger.info(f"Using device: {actual_device}")
    
    # Create generator for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for deterministic evaluation
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn_eval,
    )
    
    all_image_embeddings = []
    all_query_embeddings = []
    all_target_embeddings = []
    
    logger.info(f"Computing embeddings for {len(dataset)} samples...")
    
    for batch in tqdm(dataloader, desc="Encoding"):
        images, queries, targets, uuids = batch
        
        images = images.to(actual_device)
        

        image_features = model.get_image_features(pixel_values=images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
        all_image_embeddings.append(image_features.cpu().numpy())
        
        # Encode query texts
        text_inputs = preprocess(
                                text=queries,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=77,   # CLIP 标准长度
                            )
        text_inputs = {k: v.to(actual_device) for k, v in text_inputs.items()}

        query_features = model.get_text_features(**text_inputs)
        query_features = query_features / query_features.norm(dim=-1, keepdim=True)
        
        all_query_embeddings.append(query_features.cpu().numpy())
        
        # Encode target texts
        text_inputs = preprocess(
                                text=targets,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=77,
                            )
        text_inputs = {k: v.to(actual_device) for k, v in text_inputs.items()}
        target_features = model.get_text_features(**text_inputs)
        target_features = target_features / target_features.norm(dim=-1, keepdim=True)
        
        all_target_embeddings.append(target_features.cpu().numpy())
    
    # Concatenate embeddings
    image_embeddings = np.concatenate(all_image_embeddings, axis=0)
    query_embeddings = np.concatenate(all_query_embeddings, axis=0)
    target_embeddings = np.concatenate(all_target_embeddings, axis=0)
    
    logger.info(f"Image embeddings: {image_embeddings.shape}")
    logger.info(f"Query embeddings: {query_embeddings.shape}")
    logger.info(f"Target embeddings: {target_embeddings.shape}")

    from .metrics import compute_all_retrieval_metrics
    
    metrics = compute_all_retrieval_metrics(
        query_embeddings=query_embeddings,
        target_embeddings=target_embeddings,
        image_embeddings=image_embeddings,
        tasks=tasks,
        compute_recall=compute_recall,
        compute_mrr=compute_mrr
    )
    
    return metrics


@torch.no_grad()
def evaluate_clip_model_for_training(
    model,
    dataset: CLIPEvaluationDataset,
    batch_size: int = 64,
    device: str = 'cuda',
    seed: int = 42,
    tasks: List[str] = ["T2I", "I2T", "T2T"]
) -> Dict[str, float]:
    """
    Evaluate CLIP model during training (only MRR for early stopping).
    This is faster than full evaluation.
    
    Args:
        model: CLIP model
        dataset: Evaluation dataset
        batch_size: Batch size
        device: Device
        seed: Random seed
        tasks: List of tasks to evaluate
        
    Returns:
        Dictionary with only MRR and Mean_Rank metrics
    """
    return evaluate_clip_model(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
        seed=seed,
        tasks=tasks,
        compute_recall=False,  # Skip Recall@K for speed
        compute_mrr=True
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate CLIP model')
    
    # Model
    parser.add_argument('--model_name', type=str, default='ViT-L/14',
                       choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'])
    parser.add_argument('--checkpoint', type=str, 
                       help='Path to checkpoint, if None uses pretrained model')
    
    # Data
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--texts_dir', type=str, required=True,
                       help='Directory containing query-target JSON files')
    parser.add_argument('--splits_file', type=str, default=None,
                       help='Path to saved splits JSON (if None, will generate)')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'])
    
    # Evaluation settings
    parser.add_argument('--tasks', type=str, nargs='+', 
                       default=['T2I', 'I2T', 'T2T'],
                       choices=['T2I', 'I2T', 'T2T'],
                       help='Tasks to evaluate')
    parser.add_argument('--mrr_only', action='store_true',
                       help='Only compute MRR (faster, for training validation)')
    
    # System
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Output
    parser.add_argument('--output_file', type=str, required=True)
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    log_file = output_dir / "evaluation.log"
    setup_logger(__name__, str(log_file))
    
    logger.info("="*80)
    logger.info("CLIP Model Evaluation")
    logger.info("="*80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"MRR only: {args.mrr_only}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("="*80)
    

    train_uuids, val_uuids, test_uuids = load_splits_from_json(args.splits_file)

    
    split_map = {
        'train': train_uuids,
        'val': val_uuids,
        'test': test_uuids
    }
    selected_uuids = split_map[args.split]
    
    logger.info(f"Selected {len(selected_uuids)} samples from {args.split} split")
    
    # Load model
    logger.info("\nLoading model...")
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")

    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image

    print("Loading model xuemduan/reevaluate-clip2...")
    model = CLIPModel.from_pretrained("xuemduan/reevaluate-clip")
    preprocess = CLIPProcessor.from_pretrained("xuemduan/reevaluate-clip")
    # put move to device
    model.to(device)
    print("Model loaded and moved to device.")


    
    # Create dataset
    logger.info("\nCreating dataset...")
    dataset = CLIPEvaluationDataset(
        uuids=selected_uuids,
        image_folder=args.images_dir,
        text_folder=args.texts_dir,
        preprocessor=preprocess
    )
    
    # Evaluate
    metrics = evaluate_clip_model(
        model=model,
        preprocess=preprocess,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        tasks=args.tasks,
        compute_recall=True,
        compute_mrr=True
    )
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    for metric_name, value in sorted(metrics.items()):
        if 'Rank' in metric_name and 'Mean' in metric_name:
            logger.info(f"{metric_name}: {value:.2f}")
        else:
            logger.info(f"{metric_name}: {value:.2f}%")
    logger.info("="*80)
    
    # Save results
    results = {
        'model_name': args.model_name,
        'checkpoint': args.checkpoint,
        'split': args.split,
        'tasks': args.tasks,
        'num_samples': len(selected_uuids),
        'seed': args.seed,
        'metrics': metrics
    }
    
    save_metrics_to_json(results, args.output_file)
    logger.info(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()