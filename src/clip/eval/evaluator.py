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
from ..datasets.clip_dataset import CLIPEvalDataset as CLIPEvaluationDataset 
from ..datasets.clip_dataset import collate_fn_eval
from ..utils.data_utils import get_data_splits, load_splits_from_json
from ..utils.logging_utils import setup_logger, save_metrics_to_json
from .metrics import compute_all_retrieval_metrics, compute_training_metrics
from .fusion import *

logger = logging.getLogger(__name__)

import shutil, os, pathlib

pycache = pathlib.Path(__file__).parent / "eval" / "__pycache__"
if pycache.exists():
    print("Removing stale __pycache__:", pycache)
    shutil.rmtree(pycache)

def seed_worker(worker_id):
    """Worker seed function for DataLoader to ensure reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

text2sparql_results = {}
results = os.listdir("experiments/text2sparql/results")
for result_file in results:
    uuid = result_file.split(".")[0]
    with open(os.path.join("experiments/text2sparql/results", result_file), "r") as f:
        result_data = f.readlines()
        result_data = [line.strip() for line in result_data]
        text2sparql_results[uuid] = result_data


@torch.no_grad()
def evaluate_clip_model(
    model,
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
    logger.info("Mixed precision: DISABLED (for CPU/GPU consistency)")
    
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
        # worker_init_fn=seed_worker,
        # generator=g
    )
    
    all_image_embeddings = []
    all_query_embeddings = []
    all_target_embeddings = []
    uuid_list = []
    
    logger.info(f"Computing embeddings for {len(dataset)} samples...")
    
    for batch in tqdm(dataloader, desc="Encoding"):
        images, queries, targets, uuids = batch
        
        images = images.to(actual_device)

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
    
        uuid_list.append(uuids)

    # Concatenate embeddings
    image_embeddings = np.concatenate(all_image_embeddings, axis=0)
    query_embeddings = np.concatenate(all_query_embeddings, axis=0)
    target_embeddings = np.concatenate(all_target_embeddings, axis=0)
    uuid_list = np.concatenate(uuid_list, axis=0)


    logger.info(f"Image embeddings: {image_embeddings.shape}")
    logger.info(f"Query embeddings: {query_embeddings.shape}")
    logger.info(f"Target embeddings: {target_embeddings.shape}")
    
    # Compute metrics using realistic scenario:
    # T2I: Query → Image
    # I2T: Image → Target
    # T2T: Query → Target
    from .metrics import compute_all_retrieval_metrics
    
    metrics = compute_all_retrieval_metrics(
        query_embeddings=query_embeddings,
        target_embeddings=target_embeddings,
        image_embeddings=image_embeddings,
        tasks=tasks,
        compute_recall=compute_recall,
        compute_mrr=compute_mrr
    )
    print("0.5 0.5=============================")
    t2i_weight = 0.5
    t2t_weight = 0.5
    t2i_similarity_matrix = query_embeddings @ image_embeddings.T  # (N, N)
    t2t_similarity_matrix = query_embeddings @ target_embeddings.T  # (N, N
    similarity_matrix = (t2i_weight * t2i_similarity_matrix) + (t2t_weight * t2t_similarity_matrix)
    print("T2I")
    results = evaluate_retrieval(t2i_similarity_matrix)
    print("T2T")
    results = evaluate_retrieval(t2t_similarity_matrix)
    print("Fused")
    results = evaluate_retrieval(similarity_matrix)
    alphas = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    for alpha in alphas:
        print(f"Testing Weighted Fusion with alpha={alpha}...")
        fused_weighted = fuse_clip_and_text2sparql(
            clip_similarity_matrix=similarity_matrix,
            text2sparql_results=text2sparql_results,
            query_uuids=uuid_list,
            artefact_uuids=uuid_list,
            fusion_strategy="weighted",
            fusion_params={"alpha": alpha, "sparql_weight": 1 - alpha}
        )
        print(f"Fused matrix shape: {fused_weighted.shape}")
        print(f"Fused matrix range: [{fused_weighted.min():.4f}, {fused_weighted.max():.4f}]")
        results = evaluate_retrieval(fused_weighted)

    print("0.9 0.1=============================")
    t2i_weight = 0.1
    t2t_weight = 0.9
    t2i_similarity_matrix = query_embeddings @ image_embeddings.T  # (N, N)
    t2t_similarity_matrix = query_embeddings @ target_embeddings.T  # (N, N
    similarity_matrix = (t2i_weight * t2i_similarity_matrix) + (t2t_weight * t2t_similarity_matrix)
    print("T2I")
    results = evaluate_retrieval(t2i_similarity_matrix)
    print("T2T")
    results = evaluate_retrieval(t2t_similarity_matrix)
    print("Fused")
    results = evaluate_retrieval(similarity_matrix)
    alphas = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    for alpha in alphas:
        print(f"Testing Weighted Fusion with alpha={alpha}...")
        fused_weighted = fuse_clip_and_text2sparql(
            clip_similarity_matrix=similarity_matrix,
            text2sparql_results=text2sparql_results,
            query_uuids=uuid_list,
            artefact_uuids=uuid_list,
            fusion_strategy="weighted",
            fusion_params={"alpha": alpha, "sparql_weight": 1 - alpha}
        )
        print(f"Fused matrix shape: {fused_weighted.shape}")
        print(f"Fused matrix range: [{fused_weighted.min():.4f}, {fused_weighted.max():.4f}]")
        results = evaluate_retrieval(fused_weighted)


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
    logger.info("CLIP Model Evaluation (CPU/GPU Consistent)")
    logger.info("="*80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"MRR only: {args.mrr_only}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("="*80)
    
    # # Get data splits
    # if args.splits_file and Path(args.splits_file).exists():
    #     logger.info(f"\nLoading splits from {args.splits_file}")
    train_uuids, val_uuids, test_uuids = load_splits_from_json(args.splits_file)
    # else:
    #     logger.info("\nGenerating data splits...")
    #     train_uuids, val_uuids, test_uuids = get_data_splits(
    #         args.images_dir,
    #         args.texts_dir,
    #         test_size=0.15,
    #         val_size=0.1,
    #         random_seed=args.seed
    #     )
    test_uuids = test_uuids  # --- TEMPORARY: LIMIT TO 500 SAMPLES FOR QUICK TESTING ---
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
    print(f"Using device: {device}")
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
    
    model, preprocess = load_clip_model(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        device=device
    )
    
    # Create dataset
    logger.info("\nCreating dataset...")
    dataset = CLIPEvaluationDataset(
        uuids=selected_uuids,
        image_folder=args.images_dir,
        text_folder=args.texts_dir,
        preprocessor=preprocess
    )
    
    # Evaluate
    logger.info("\nEvaluating...")
    if args.mrr_only:
        metrics = evaluate_clip_model_for_training(
            model=model,
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            tasks=args.tasks
        )
    else:
        metrics = evaluate_clip_model(
            model=model,
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