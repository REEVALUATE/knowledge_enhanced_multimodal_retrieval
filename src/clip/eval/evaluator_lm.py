"""
Evaluate text-only models (MPNet, E5, GTE) on text-to-text retrieval.
"""

import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.clip.utils.data_utils import get_data_splits, load_splits_from_json
from src.clip.utils.logging_utils import setup_logger, save_metrics_to_json

logger = logging.getLogger(__name__)


class TextOnlyDataset(Dataset):
    """Dataset for text-only models - always loads all 5 variants."""
    
    def __init__(
        self,
        uuids: List[str],
        text_folder: str,
        description_type: str,
        random_seed: int = 42,
        num_variants: int = 5
    ):
        self.uuids = uuids
        self.text_folder = Path(text_folder)
        self.description_type = description_type
        self.random_seed = random_seed
        self.num_variants = num_variants
        
        self.desc_key_map = {
            'content': 'content_descriptions',
            'metadata': 'metadata_descriptions',
            'hybrid_o1': 'hybrid_descriptions',
            'hybrid_o2': 'hybrid_descriptions'
        }
        
        logger.info(f"Text-only dataset: {len(uuids)} samples, loading all {num_variants} variants")
    
    def __len__(self):
        return len(self.uuids)
    
    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        
        text_path = self.text_folder / f"{uuid}.json"
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                desc_key = self.desc_key_map[self.description_type]
                descriptions = data.get(desc_key, [])
                
                texts = []
                for i in range(self.num_variants):
                    if i < len(descriptions) and descriptions[i].strip():
                        texts.append(descriptions[i])
                    else:
                        texts.append("")
                
                while len(texts) < self.num_variants:
                    texts.append("")
                    
        except Exception as e:
            logger.error(f"Error loading text for {uuid}: {e}")
            texts = [""] * self.num_variants
        
        return texts


def collate_fn(batch):
    """Collate function - returns list of text lists."""
    return list(batch)


def seed_worker(worker_id):
    """Worker seed function for DataLoader to ensure reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def evaluate_text_model(
    model: SentenceTransformer,
    dataset: TextOnlyDataset,
    batch_size: int = 32,
    device: str = 'cuda',
    mode: str = 'multi',
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate text-only model on text-to-text retrieval.
    
    Single mode: variant 0 finds variants 1-4 (N queries → N×4 candidates)
    Multi mode: each variant finds other 4, then average (5× N queries → N×4 candidates each)
    
    Args:
        model: Sentence transformer model
        dataset: Text dataset
        batch_size: Batch size
        device: Device
        mode: 'single' or 'multi'
        seed: Random seed
        
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
        shuffle=True,  # Shuffle with fixed seed
        num_workers=4,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    all_text_embeddings_by_variant = [[] for _ in range(5)]
    
    logger.info(f"Encoding texts for {len(dataset)} samples (mode={mode})...")
    
    for batch in tqdm(dataloader, desc="Encoding texts"):
        # batch is list of text lists (each with 5 variants)
        batch_size_actual = len(batch)
        for v_idx in range(5):
            texts = [batch[i][v_idx] for i in range(batch_size_actual)]
            embeddings = model.encode(
                texts,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize during encoding
            )
            all_text_embeddings_by_variant[v_idx].append(embeddings.cpu().numpy())
    
    # Concatenate embeddings
    text_embeddings_by_variant = []
    for v_idx in range(5):
        text_emb = np.concatenate(all_text_embeddings_by_variant[v_idx], axis=0)
        text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
        text_embeddings_by_variant.append(text_emb)
    
    N = len(text_embeddings_by_variant[0])
    logger.info(f"Text embeddings per variant: {text_embeddings_by_variant[0].shape}")
    
    # Compute T2T metrics
    k_values = [1, 5, 10, 20]
    metrics = {}
    
    if mode == 'single':
        # Single mode: variant 0 finds variants 1-4
        # 3. Text-to-Text (T2T): Variant 0 finds other 4 variants
        logger.info("Computing T2T metrics (variant 0 → variants 1-4)...")
        
        # Query: variant 0
        query_embeddings = text_embeddings_by_variant[0]  # (N, D)
        
        # Candidates: variants 1-4
        candidate_texts = []
        text_to_artifact = []
        
        for artifact_idx in range(N):
            for v_idx in range(1, 5):  # Only variants 1-4
                candidate_texts.append(text_embeddings_by_variant[v_idx][artifact_idx])
                text_to_artifact.append(artifact_idx)
        
        candidate_texts = np.array(candidate_texts)  # (N×4, D)
        text_to_artifact = np.array(text_to_artifact)  # (N×4,)
        
        # Compute similarity: (N, N×4)
        t2t_similarity = query_embeddings @ candidate_texts.T
        
        # Recall@K
        for k in k_values:
            correct_count = 0
            for i in range(N):
                top_k_indices = np.argsort(-t2t_similarity[i])[:k]
                top_k_artifacts = text_to_artifact[top_k_indices]
                
                if i in top_k_artifacts:
                    correct_count += 1
            
            recall = correct_count / N * 100
            metrics[f'T2T_R@{k}'] = recall
        
        # MRR
        reciprocal_ranks = []
        for i in range(N):
            ranking = np.argsort(-t2t_similarity[i])
            ranked_artifacts = text_to_artifact[ranking]
            
            # Find first occurrence of correct artifact
            position = np.where(ranked_artifacts == i)[0][0] + 1
            reciprocal_ranks.append(1.0 / position)
        
        metrics['T2T_MRR'] = np.mean(reciprocal_ranks) * 100
        
        # Mean Rank
        ranks = []
        for i in range(N):
            ranking = np.argsort(-t2t_similarity[i])
            ranked_artifacts = text_to_artifact[ranking]
            position = np.where(ranked_artifacts == i)[0][0] + 1
            ranks.append(position)
        
        metrics['T2T_Mean_Rank'] = np.mean(ranks)
        
        logger.info(f"T2T candidate pool size: {len(candidate_texts)} (N×4 = {N}×4)")
        
    else:
        logger.info("Computing T2T metrics (each variant as query, excluding self)...")
        
        # Build candidate pool with all 5 variants
        all_texts_flat_t2t = []
        text_to_artifact_t2t = []
        variant_indices = []
        
        for artifact_idx in range(N):
            for v_idx in range(5):
                all_texts_flat_t2t.append(text_embeddings_by_variant[v_idx][artifact_idx])
                text_to_artifact_t2t.append(artifact_idx)
                variant_indices.append(v_idx)
        
        all_texts_flat_t2t = np.array(all_texts_flat_t2t)  # (N×5, D)
        text_to_artifact_t2t = np.array(text_to_artifact_t2t)  # (N×5,)
        variant_indices = np.array(variant_indices)  # (N×5,)
        
        # For each variant as query
        all_recalls = {k: [] for k in k_values}
        all_reciprocal_ranks = []
        all_ranks = []
        
        for query_v_idx in range(5):
            query_embeddings = text_embeddings_by_variant[query_v_idx]  # (N, D)
            
            # Compute similarity to all candidates
            t2t_similarity = query_embeddings @ all_texts_flat_t2t.T  # (N, N×5)
            
            # For each query sample, exclude self-match
            for i in range(N):
                # Find the index in flat array that corresponds to same artifact and same variant
                self_match_idx = i * 5 + query_v_idx
                
                # Mask out self-match
                t2t_similarity_masked = t2t_similarity[i].copy()
                t2t_similarity_masked[self_match_idx] = -np.inf
                
                # Get ranking after masking
                ranking = np.argsort(-t2t_similarity_masked)
                ranked_artifacts = text_to_artifact_t2t[ranking]
                
                # For Recall@K
                for k in k_values:
                    top_k_artifacts = ranked_artifacts[:k]
                    if i in top_k_artifacts:
                        all_recalls[k].append(1.0)
                    else:
                        all_recalls[k].append(0.0)
                
                # For MRR and Mean Rank
                position = np.where(ranked_artifacts == i)[0][0] + 1
                all_reciprocal_ranks.append(1.0 / position)
                all_ranks.append(position)
        
        # Average across all query variants
        for k in k_values:
            metrics[f'T2T_R@{k}'] = np.mean(all_recalls[k]) * 100
        
        metrics['T2T_MRR'] = np.mean(all_reciprocal_ranks) * 100
        metrics['T2T_Mean_Rank'] = np.mean(all_ranks)
        
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate text-only models')
    
    # Model
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name: sentence-transformers/all-mpnet-base-v2, '
                            'intfloat/e5-base-v2, thenlper/gte-large')
    
    # Data
    parser.add_argument('--texts_dir', type=str, required=True)
    parser.add_argument('--description_type', type=str, required=True,
                       choices=['content', 'metadata', 'hybrid_o1', 'hybrid_o2'])
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Needed for data split generation')
    parser.add_argument('--splits_file', type=str, default=None)
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'])
    
    # Evaluation mode
    parser.add_argument('--eval_mode', type=str, default='multi',
                       choices=['single', 'multi'])
    
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
    logger.info(f"Description type: {args.description_type}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Evaluation mode: {args.eval_mode}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("="*80)
    
    # Get data splits
    if args.splits_file and Path(args.splits_file).exists():
        logger.info(f"\nLoading splits from {args.splits_file}")
        train_uuids, val_uuids, test_uuids = load_splits_from_json(args.splits_file)
    else:
        logger.info("\nGenerating data splits...")
        train_uuids, val_uuids, test_uuids = get_data_splits(
            args.images_dir,
            args.texts_dir,
            test_size=0.15,
            val_size=0.1,
            random_seed=args.seed
        )
    
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
    dataset = TextOnlyDataset(
        uuids=selected_uuids,
        text_folder=args.texts_dir,
        description_type=args.description_type,
        random_seed=args.seed
    )
    
    # Evaluate
    logger.info("\nEvaluating...")
    metrics = evaluate_text_model(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        mode=args.eval_mode,
        seed=args.seed
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
        'description_type': args.description_type,
        'split': args.split,
        'eval_mode': args.eval_mode,
        'num_samples': len(selected_uuids),
        'seed': args.seed,
        'metrics': metrics
    }
    
    save_metrics_to_json(results, args.output_file)
    logger.info(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
