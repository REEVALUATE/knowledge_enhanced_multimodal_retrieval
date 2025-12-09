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
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip

from ..model.clip_model import load_clip_model
from ..datasets.clip_dataset import CLIPEvalDatasetHF as CLIPEvaluationDataset 
from ..datasets.clip_dataset import collate_fn_eval
from ..utils.data_utils import get_data_splits, load_splits_from_json
from ..utils.logging_utils import setup_logger, save_metrics_to_json
from .metrics import compute_all_retrieval_metrics, compute_training_metrics
from datasets import load_dataset

logger = logging.getLogger(__name__)


def seed_worker(worker_id):
    """Worker seed function for DataLoader to ensure reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def evaluate_clip_model(
    model,
    dataset: CLIPEvaluationDataset,
    batch_size: int = 64,
    device: str = 'cuda',
    seed: int = 42,
    tasks: List[str] = ["T2I", "I2T", "T2T"],
    compute_recall: bool = True,
    compute_mrr: bool = True,
    t2i_weight: float = 0.5,
    t2t_weight: float = 0.5
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
        num_workers=4,
        pin_memory=use_cuda,
        collate_fn=collate_fn_eval,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    all_image_embeddings = []
    all_query_embeddings = []
    all_target_embeddings = []
    
    logger.info(f"Computing embeddings for {len(dataset)} samples...")
    
    for batch in tqdm(dataloader, desc="Encoding"):
        images, queries, targets = batch
        
        images = images.to(actual_device)
        
        # ✅ FIXED: Remove mixed precision, use float32 consistently
        # Encode images
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        all_image_embeddings.append(image_features.cpu().numpy())
        
        # Encode query texts
        query_tokens = clip.tokenize(queries, truncate=True).to(actual_device)
        query_features = model.encode_text(query_tokens)
        query_features = query_features / query_features.norm(dim=-1, keepdim=True)
        all_query_embeddings.append(query_features.cpu().numpy())
        
        # Encode target texts
        target_tokens = clip.tokenize(targets, truncate=True).to(actual_device)
        target_features = model.encode_text(target_tokens)
        target_features = target_features / target_features.norm(dim=-1, keepdim=True)
        all_target_embeddings.append(target_features.cpu().numpy())
    
    # Concatenate embeddings
    image_embeddings = np.concatenate(all_image_embeddings, axis=0)
    query_embeddings = np.concatenate(all_query_embeddings, axis=0)
    target_embeddings = np.concatenate(all_target_embeddings, axis=0)
    
    logger.info(f"Image embeddings: {image_embeddings.shape}")
    logger.info(f"Query embeddings: {query_embeddings.shape}")
    logger.info(f"Target embeddings: {target_embeddings.shape}")
    
    # Compute metrics using realistic scenario:
    # T2I: Query → Image
    # I2T: Image → Target
    # T2T: Query → Target
    from .metrics import compute_retrieval_metrics_final
    
    metrics = compute_retrieval_metrics_final(
        query_embeddings=query_embeddings,
        target_embeddings=target_embeddings,
        image_embeddings=image_embeddings,
        t2i_weight=t2i_weight,
        t2t_weight=t2t_weight,
    )
    
    return metrics



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
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'])
    
    parser.add_argument('--t2i_weight', type=float, default=0.5,
                          help='Weight for T2I task in final score computation')
    parser.add_argument('--t2t_weight', type=float, default=0.5,
                            help='Weight for T2T task in final score computation')

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
    
    # Load model
    logger.info("\nLoading model...")
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
    
    model, preprocess = load_clip_model(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        device=device
    )
    
    logger.info("\nCreating dataset...")
    ds = load_dataset("xuemduan/reevaluate-image-text-pairs")

    dataset = CLIPEvaluationDataset(
        hf_dataset=ds[args.split],
        preprocessor=preprocess
    )
    
    # Evaluate
    logger.info("\nEvaluating...")

    metrics = evaluate_clip_model(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        tasks=args.tasks,
        compute_recall=True,
        compute_mrr=True,
        t2i_weight=args.t2i_weight,
        t2t_weight=args.t2t_weight
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
        'num_samples': len(dataset),
        'seed': args.seed,
        'metrics': metrics
    }
    
    save_metrics_to_json(results, args.output_file)
    logger.info(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()