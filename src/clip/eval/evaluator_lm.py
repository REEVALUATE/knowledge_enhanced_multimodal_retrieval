"""
Evaluate text-only models (MPNet, E5, GTE) on text-to-text retrieval.
Uses unified metrics from metrics.py.
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
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ..datasets.clip_dataset import TextOnlyDataset as CLIPEvaluationDataset 
from ..datasets.clip_dataset import collate_fn_eval_texts
from src.clip.utils.data_utils import get_data_splits, load_splits_from_json
from src.clip.utils.logging_utils import setup_logger, save_metrics_to_json
from src.clip.eval.metrics import compute_retrieval_metrics

logger = logging.getLogger(__name__)


def seed_worker(worker_id):
    """Worker seed function for DataLoader to ensure reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def evaluate_text_model(
    model: SentenceTransformer,
    dataset: CLIPEvaluationDataset,
    batch_size: int = 32,
    device: str = 'cuda',
    seed: int = 42,
    compute_recall: bool = True,
    compute_mrr: bool = True
) -> Dict[str, float]:
    """
    Evaluate text-only model on text-to-text retrieval.
    Query (mixed) → Target (hybrid) retrieval.
    
    Args:
        model: Sentence transformer model
        dataset: Text dataset
        batch_size: Batch size
        device: Device
        seed: Random seed
        compute_recall: Whether to compute Recall@K
        compute_mrr: Whether to compute MRR and Mean Rank
        
    Returns:
        Dictionary of T2T metrics only
    """
    model = model.to(device)
    model.eval()
    
    # Create generator for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for deterministic evaluation
        num_workers=4,
        collate_fn=collate_fn_eval_texts,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    all_query_embeddings = []
    all_target_embeddings = []
    
    logger.info(f"Encoding texts for {len(dataset)} samples...")
    
    for batch in tqdm(dataloader, desc="Encoding Texts"):
        queries, targets = batch
        
        # Encode queries
        query_embeds = model.encode(
            queries,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        all_query_embeddings.append(query_embeds.cpu().numpy())
        
        # Encode targets
        target_embeds = model.encode(
            targets,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        all_target_embeddings.append(target_embeds.cpu().numpy())
    
    # Concatenate embeddings
    query_embeddings = np.concatenate(all_query_embeddings, axis=0)
    target_embeddings = np.concatenate(all_target_embeddings, axis=0)
    
    # Normalize (redundant if normalize_embeddings=True, but ensures it)
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    target_embeddings = target_embeddings / np.linalg.norm(target_embeddings, axis=1, keepdims=True)
    
    logger.info(f"Query embeddings: {query_embeddings.shape}")
    logger.info(f"Target embeddings: {target_embeddings.shape}")
    
    # Compute T2T metrics using unified function
    logger.info("Computing T2T metrics (query → target retrieval)...")
    metrics = compute_retrieval_metrics(
        query_embeddings=query_embeddings,
        candidate_embeddings=target_embeddings,
        prefix="T2T",
        compute_recall=compute_recall,
        compute_mrr=compute_mrr
    )
    
    return metrics


@torch.no_grad()
def evaluate_text_model_for_training(
    model: SentenceTransformer,
    dataset: CLIPEvaluationDataset,
    batch_size: int = 32,
    device: str = 'cuda',
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate text model during training (only MRR for early stopping).
    This is faster than full evaluation.
    
    Args:
        model: Sentence transformer model
        dataset: Text dataset
        batch_size: Batch size
        device: Device
        seed: Random seed
        
    Returns:
        Dictionary with only MRR and Mean_Rank metrics
    """
    return evaluate_text_model(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
        seed=seed,
        compute_recall=False,  # Skip Recall@K for speed
        compute_mrr=True
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate text-only models')
    
    # Model
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name: sentence-transformers/all-mpnet-base-v2, '
                            'intfloat/e5-base-v2, thenlper/gte-large')
    
    # Data
    parser.add_argument('--texts_dir', type=str, required=True,
                       help='Directory containing query-target JSON files')
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Needed for data split generation')
    parser.add_argument('--splits_file', type=str, default=None)
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'])
    
    # Evaluation settings
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
    logger.info("Text-Only Model Evaluation (T2T only)")
    logger.info("="*80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Split: {args.split}")
    logger.info(f"MRR only: {args.mrr_only}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("="*80)
    
    # Get data splits
    logger.info(f"\nLoading splits from {args.splits_file}")
    train_uuids, val_uuids, test_uuids = load_splits_from_json(args.splits_file)

    split_map = {
        'train': train_uuids,
        'val': val_uuids,
        'test': test_uuids
    }
    selected_uuids = split_map[args.split]
    
    logger.info(f"Selected {len(selected_uuids)} samples from {args.split} split")
    
    # Load model
    logger.info(f"\nLoading model: {args.model_name}")
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
    
    model = SentenceTransformer(args.model_name, device=device)
    
    # Create dataset
    logger.info("\nCreating dataset...")
    # dataset = TextOnlyDataset(
    #     uuids=selected_uuids,
    #     text_folder=args.texts_dir
    # )
    dataset = CLIPEvaluationDataset(
        uuids=selected_uuids,
        text_folder=args.texts_dir
    )
    
    # Evaluate
    logger.info("\nEvaluating...")
    if args.mrr_only:
        metrics = evaluate_text_model_for_training(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed
        )
    else:
        metrics = evaluate_text_model(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            compute_recall=True,
            compute_mrr=True
        )
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS (T2T Only)")
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
        'split': args.split,
        'num_samples': len(selected_uuids),
        'seed': args.seed,
        'metrics': metrics
    }
    
    save_metrics_to_json(results, args.output_file)
    logger.info(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()