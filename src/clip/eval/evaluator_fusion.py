"""
Evaluator for Fusion Models (Stage 2) 
Computes final artifact retrieval metrics by fusing T2I and T2T scores.
"""

import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import clip

from ..utils.data_utils import load_splits_from_json
from ..model.clip_model import load_clip_model
from ..datasets.clip_dataset import CLIPEvalDatasetHF, collate_fn_eval
from ..model.fusion_model import FusionModel
from datasets import load_dataset

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_fusion_model(
    fusion_model: FusionModel,
    dataset: CLIPEvalDatasetHF,
    batch_size: int = 64,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Evaluate fusion model on final artifact retrieval."""
    fusion_model.eval()
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_eval
    )
    
    # Encode all
    all_image_embeds = []
    all_query_embeds = []
    all_target_embeds = []
    
    logger.info("Encoding all samples...")
    for images, queries, targets in tqdm(loader, desc="Encoding"):
        images = images.to(device)
        query_tokens = clip.tokenize(queries, truncate=True).to(device)
        target_tokens = clip.tokenize(targets, truncate=True).to(device)
        
        image_embeds = fusion_model.encode_image(images)
        query_embeds = fusion_model.encode_query(query_tokens)
        target_embeds = fusion_model.encode_target(target_tokens)
        
        all_image_embeds.append(image_embeds.cpu())
        all_query_embeds.append(query_embeds.cpu())
        all_target_embeds.append(target_embeds.cpu())
    
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_query_embeds = torch.cat(all_query_embeds, dim=0)
    all_target_embeds = torch.cat(all_target_embeds, dim=0)
    
    N = len(all_image_embeds)
    logger.info(f"Total samples: {N}")
    
    # Compute fused similarity matrix in blocks
    logger.info("Computing fused scores...")
    
    query_batch_size = 50
    cand_batch_size = 500
    
    fused_sim_matrix = np.zeros((N, N), dtype=np.float32)
    
    first_block = True  # ✅ For debugging
    
    for qi in tqdm(range(0, N, query_batch_size), desc="Query batches"):
        qe = min(qi + query_batch_size, N)
        batch_queries = all_query_embeds[qi:qe].to(device)
        
        for ci in range(0, N, cand_batch_size):
            ce = min(ci + cand_batch_size, N)
            batch_images = all_image_embeds[ci:ce].to(device)
            batch_targets = all_target_embeds[ci:ce].to(device)
            
            # Compute block
            block_scores = fusion_model(
                query_embed=batch_queries,
                image_embed=batch_images,
                target_embed=batch_targets
            )
            
            if first_block:
                logger.info("="*60)
                logger.info("SCORE STATISTICS (First Block)")
                logger.info("="*60)
                logger.info(f"Fusion scores:")
                logger.info(f"  Range: [{block_scores.min().item():.4f}, {block_scores.max().item():.4f}]")
                logger.info(f"  Mean: {block_scores.mean().item():.4f}, Std: {block_scores.std().item():.4f}")
                
                # Compare with baseline
                t2i = batch_queries @ batch_images.T
                t2t = batch_queries @ batch_targets.T
                baseline = (t2i + t2t) / 2
                logger.info(f"Baseline (0.5*T2I + 0.5*T2T):")
                logger.info(f"  Range: [{baseline.min().item():.4f}, {baseline.max().item():.4f}]")
                logger.info(f"  Mean: {baseline.mean().item():.4f}, Std: {baseline.std().item():.4f}")
                logger.info("="*60)
                first_block = False
            
            # Store in matrix
            fused_sim_matrix[qi:qe, ci:ce] = block_scores.cpu().numpy()
            
            del batch_images, batch_targets, block_scores
            torch.cuda.empty_cache()
    
    # Compute metrics
    from ..eval.metrics import compute_retrieval_metrics_fusion
    
    metrics = compute_retrieval_metrics_fusion(
        similarity_matrix=fused_sim_matrix,
        prefix="",
        k_values=[1, 5, 10, 20],
        compute_recall=True,
        compute_mrr=True
    )
    
    logger.info("="*60)
    logger.info("Fusion Model Evaluation Results")
    logger.info("="*60)
    for k, v in metrics.items():
        if 'R@' in k or 'MRR' in k:
            logger.info(f"{k}: {v:.2f}%")
        else:
            logger.info(f"{k}: {v:.2f}")
    logger.info("="*60)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Fusion Model')
    
    parser.add_argument('--model_name', type=str, default='ViT-L/14')
    parser.add_argument('--clip_checkpoint', type=str, required=True)
    parser.add_argument('--fusion_checkpoint', type=str, required=True)
    parser.add_argument('--fusion_type', type=str, required=True,
                       choices=['linear', 'cross_attention', 'gated', 'bilinear'])
    
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--texts_dir', type=str, required=True)
    parser.add_argument('--splits_file', type=str, required=True)
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'])
    parser.add_argument('--max_text_length', type=int, default=150)
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_file', type=str, default=None)
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    train_uuids, val_uuids, test_uuids = load_splits_from_json(args.splits_file)
    
    if args.split == 'train':
        eval_uuids = train_uuids
    elif args.split == 'val':
        eval_uuids = val_uuids
    else:
        eval_uuids = test_uuids
    
    logger.info(f"Evaluating on {args.split} set: {len(eval_uuids)} samples")
    
    logger.info(f"Loading CLIP from {args.clip_checkpoint}")
    clip_model, preprocess = load_clip_model(
        model_name=args.model_name,
        checkpoint_path=args.clip_checkpoint,
        device=args.device
    )
    
    embed_dim = 768 if 'L/14' in args.model_name else 512
    fusion_model = FusionModel(
        clip_model=clip_model,
        fusion_type=args.fusion_type,
        embed_dim=embed_dim
    ).to(args.device)
    
    logger.info(f"Loading fusion head from {args.fusion_checkpoint}")
    checkpoint = torch.load(args.fusion_checkpoint, map_location=args.device)
    fusion_model.fusion_head.load_state_dict(checkpoint['fusion_head_state_dict'])
    
    dataset = CLIPEvalDataset(
        uuids=eval_uuids,
        image_folder=args.images_dir,
        text_folder=args.texts_dir,
        preprocessor=preprocess,
        max_text_length=args.max_text_length
    )
    
    metrics = evaluate_fusion_model(
        fusion_model=fusion_model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=args.device
    )
    
    results = {
        'model_name': args.model_name,
        'clip_checkpoint': args.clip_checkpoint,
        'fusion_checkpoint': args.fusion_checkpoint,
        'fusion_type': args.fusion_type,
        'split': args.split,
        'num_samples': len(eval_uuids),
        'metrics': metrics
    }
    
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()