"""
Unified metrics computation for retrieval tasks.
Supports T2I, I2T, and T2T retrieval evaluation.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def compute_recall_at_k(
    similarity_matrix: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """
    Compute Recall@K for a similarity matrix.
    
    Args:
        similarity_matrix: (N_queries, N_candidates) similarity scores
        k_values: List of K values for Recall@K
        
    Returns:
        Dictionary with Recall@K scores (as percentages)
    """
    N = similarity_matrix.shape[0]
    
    # Get top-k indices
    max_k = max(k_values)
    # if max_k < similarity_matrix.shape[1]:
    #     top_k_indices = np.argpartition(-similarity_matrix, max_k - 1, axis=1)[:, :max_k]
    # else:
    top_k_indices = np.argsort(-similarity_matrix, axis=1)
    
    # Ground truth: diagonal (query i matches candidate i)
    targets = np.arange(N)[:, None]  # (N, 1)
    
    recalls = {}
    for k in k_values:
        correct = np.any(top_k_indices[:, :k] == targets, axis=1)
        recalls[f"R@{k}"] = np.mean(correct) * 100.0
    
    return recalls


def compute_mrr_and_mean_rank(
    similarity_matrix: np.ndarray
) -> Dict[str, float]:
    """
    Compute Mean Reciprocal Rank (MRR) and Mean Rank.
    
    Args:
        similarity_matrix: (N_queries, N_candidates) similarity scores
        
    Returns:
        Dictionary with MRR (as percentage) and Mean_Rank
    """
    N = similarity_matrix.shape[0]
    
    # Get rankings (sorted indices by descending similarity)
    rankings = np.argsort(-similarity_matrix, axis=1)
    
    # Ground truth: diagonal
    targets = np.arange(N)[:, None]
    
    # Find position of target in ranking (0-based)
    positions = np.argmax(rankings == targets, axis=1) + 1  # Convert to 1-based
    
    mrr = np.mean(1.0 / positions) * 100.0
    mean_rank = np.mean(positions)
    
    return {
        "MRR": mrr,
        "Mean_Rank": mean_rank
    }


def compute_retrieval_metrics(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    prefix: str = "",
    k_values: List[int] = [1, 5, 10, 20],
    compute_recall: bool = True,
    compute_mrr: bool = True
) -> Dict[str, float]:
    """
    Compute retrieval metrics given query and candidate embeddings.
    
    Args:
        query_embeddings: (N, D) normalized embeddings
        candidate_embeddings: (N, D) normalized embeddings
        prefix: Metric name prefix (e.g., "T2I", "I2T", "T2T")
        k_values: List of K values for Recall@K
        compute_recall: Whether to compute Recall@K
        compute_mrr: Whether to compute MRR and Mean Rank
        
    Returns:
        Dictionary of metrics with prefix
    """
    # Compute similarity matrix
    similarity_matrix = query_embeddings @ candidate_embeddings.T  # (N, N)

    metrics = {}
    
    if compute_recall:
        recalls = compute_recall_at_k(similarity_matrix, k_values)
        for k, v in recalls.items():
            metrics[f"{prefix}_{k}" if prefix else k] = v
    
    if compute_mrr:
        mrr_metrics = compute_mrr_and_mean_rank(similarity_matrix)
        for k, v in mrr_metrics.items():
            metrics[f"{prefix}_{k}" if prefix else k] = v
    
    return metrics


def compute_retrieval_metrics_final(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    prefix: str = "",
    k_values: List[int] = [1, 5, 10, 20],
    compute_recall: bool = True,
    compute_mrr: bool = True,
    t2i_weight: float = 0.5,
    t2t_weight: float = 0.5
) -> Dict[str, float]:
    """
    Compute retrieval metrics given query and candidate embeddings.
    
    Args:
        query_embeddings: (N, D) normalized embeddings
        candidate_embeddings: (N, D) normalized embeddings
        prefix: Metric name prefix (e.g., "T2I", "I2T", "T2T")
        k_values: List of K values for Recall@K
        compute_recall: Whether to compute Recall@K
        compute_mrr: Whether to compute MRR and Mean Rank
        
    Returns:
        Dictionary of metrics with prefix
    """
    # Compute similarity matrix
    t2i_similarity_matrix = query_embeddings @ image_embeddings.T  # (N, N)
    t2t_similarity_matrix = query_embeddings @ target_embeddings.T  # (N, N
    print("compute_retrieval_metrics_final:", t2i_weight, t2t_weight)
    similarity_matrix = (t2i_weight * t2i_similarity_matrix) + (t2t_weight * t2t_similarity_matrix)
    
    metrics = {}
    
    if compute_recall:
        recalls = compute_recall_at_k(similarity_matrix, k_values)
        for k, v in recalls.items():
            metrics[f"{prefix}_{k}" if prefix else k] = v
    
    if compute_mrr:
        mrr_metrics = compute_mrr_and_mean_rank(similarity_matrix)
        for k, v in mrr_metrics.items():
            metrics[f"{prefix}_{k}" if prefix else k] = v
    
    return metrics


def compute_retrieval_metrics_fusion(
    similarity_matrix: np.ndarray,
    prefix: str = "",
    k_values: List[int] = [1, 5, 10, 20],
    compute_recall: bool = True,
    compute_mrr: bool = True
) -> Dict[str, float]:
    
    metrics = {}
    
    if compute_recall:
        recalls = compute_recall_at_k(similarity_matrix, k_values)
        for k, v in recalls.items():
            metrics[f"{prefix}_{k}" if prefix else k] = v
    
    if compute_mrr:
        mrr_metrics = compute_mrr_and_mean_rank(similarity_matrix)
        for k, v in mrr_metrics.items():
            metrics[f"{prefix}_{k}" if prefix else k] = v
    
    return metrics


def compute_all_retrieval_metrics(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    k_values: List[int] = [1, 5, 10, 20],
    tasks: List[str] = ["T2I", "I2T", "T2T"],
    compute_recall: bool = True,
    compute_mrr: bool = True
) -> Dict[str, float]:
    """
    Compute all retrieval metrics (T2I, I2T, T2T) using realistic scenario.
    
    Realistic scenario:
    - T2I: Query → Image (user query finds images)
    - I2T: Image → Target (image finds complete descriptions)
    - T2T: Query → Target (user query finds complete descriptions)
    
    Args:
        query_embeddings: (N, D) normalized query text embeddings (incomplete/mixed)
        target_embeddings: (N, D) normalized target text embeddings (complete/hybrid)
        image_embeddings: (N, D) normalized image embeddings
        k_values: List of K values for Recall@K
        tasks: List of tasks to evaluate (subset of ["T2I", "I2T", "T2T"])
        compute_recall: Whether to compute Recall@K
        compute_mrr: Whether to compute MRR and Mean Rank
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    if "T2I" in tasks:
        # Text-to-Image: Query → Image (user query finds images)
        t2i_metrics = compute_retrieval_metrics(
            query_embeddings, image_embeddings,
            prefix="T2I",
            k_values=k_values,
            compute_recall=compute_recall,
            compute_mrr=compute_mrr
        )
        metrics.update(t2i_metrics)
    
    if "I2T" in tasks:
        # Image-to-Text: Image → Target (image finds complete descriptions)
        i2t_metrics = compute_retrieval_metrics(
            image_embeddings, target_embeddings,
            prefix="I2T",
            k_values=k_values,
            compute_recall=compute_recall,
            compute_mrr=compute_mrr
        )
        metrics.update(i2t_metrics)
    
    if "T2T" in tasks:
        # Text-to-Text: Query → Target (user query finds complete descriptions)
        t2t_metrics = compute_retrieval_metrics(
            query_embeddings, target_embeddings,
            prefix="T2T",
            k_values=k_values,
            compute_recall=compute_recall,
            compute_mrr=compute_mrr
        )
        metrics.update(t2t_metrics)
    
    return metrics



def compute_training_metrics(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    tasks: List[str] = ["T2I", "I2T", "T2T"]
) -> Dict[str, float]:
    """
    Compute only MRR metrics for training (early stopping).
    This is faster than computing all metrics.
    
    Args:
        query_embeddings: (N, D) normalized query text embeddings
        target_embeddings: (N, D) normalized target text embeddings
        image_embeddings: (N, D) normalized image embeddings
        tasks: List of tasks to evaluate
        
    Returns:
        Dictionary with only MRR and Mean_Rank metrics
    """
    return compute_all_retrieval_metrics(
        query_embeddings,
        target_embeddings,
        image_embeddings,
        tasks=tasks,
        compute_recall=False,
        compute_mrr=True
    )


def compute_metrics_multi_mode(
    image_embeddings: np.ndarray,
    text_embeddings_by_variant: List[np.ndarray]
) -> Dict[str, float]:

    logger.warning(
        "compute_metrics_multi_mode is DEPRECATED. "
        "Use compute_all_retrieval_metrics with query_embeddings and target_embeddings instead."
    )
    
    # Just use the first (and only) variant
    text_embeddings = text_embeddings_by_variant[0]
    
    # Old behavior: text → text (unrealistic)
    return compute_all_retrieval_metrics(
        query_embeddings=text_embeddings,
        target_embeddings=text_embeddings,  # Self-retrieval (unrealistic)
        image_embeddings=image_embeddings,
        tasks=["T2I", "I2T", "T2T"]
    )


def compute_metrics_single_4train(
    image_embeddings: np.ndarray,
    text_embeddings_by_variant: List[np.ndarray]
) -> Dict[str, float]:
    """
    DEPRECATED: For backward compatibility.
    
    WARNING: Use compute_training_metrics instead.
    """
    logger.warning(
        "compute_metrics_single_4train is DEPRECATED. "
        "Use compute_training_metrics with separate query and target embeddings."
    )
    
    text_embeddings = text_embeddings_by_variant[0]
    
    return compute_training_metrics(
        query_embeddings=text_embeddings,
        target_embeddings=text_embeddings,  # Self-retrieval
        image_embeddings=image_embeddings,
        tasks=["T2I", "I2T", "T2T"]
    )


def compute_metrics_multi_4train(
    image_embeddings: np.ndarray,
    text_embeddings_by_variant: List[np.ndarray]
) -> Dict[str, float]:
    """
    DEPRECATED: For backward compatibility.
    
    WARNING: Use compute_training_metrics instead.
    """
    logger.warning(
        "compute_metrics_multi_4train is DEPRECATED. "
        "Use compute_training_metrics with separate query and target embeddings."
    )
    
    text_embeddings = text_embeddings_by_variant[0]
    
    return compute_training_metrics(
        query_embeddings=text_embeddings,
        target_embeddings=text_embeddings,  # Self-retrieval
        image_embeddings=image_embeddings,
        tasks=["T2I", "I2T", "T2T"]
    )