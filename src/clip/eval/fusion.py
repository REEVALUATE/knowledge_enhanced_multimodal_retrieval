import numpy as np
from typing import Dict, List
from .metrics import compute_recall_at_k, compute_mrr_and_mean_rank


def evaluate_retrieval(
    similarity_matrix: np.ndarray,
) -> Dict[str, float]:
    metrics = {}

    k_values = [1, 5, 10, 20]
    recalls = compute_recall_at_k(similarity_matrix, k_values)
    for k, v in recalls.items():
        metrics[k] = v

    mrr_metrics = compute_mrr_and_mean_rank(similarity_matrix)
    for k, v in mrr_metrics.items():
        metrics[k] = v
    print("evaluate_retrieval:", metrics)
    return metrics

def weighted_fusion(
    clip_similarity_matrix: np.ndarray,
    text2sparql_results: Dict[str, List[str]],
    query_uuids: List[str],
    artefact_uuids: List[str],
    alpha: float = 0.7,
    sparql_weight: float = 0.3
) -> np.ndarray:
    """
    Fuse CLIP similarity scores with Text2SPARQL binary relevance using weighted combination.
    
    Formula:
        S_final(q, d) = alpha * S_CLIP(q, d) + (1 - alpha) * I(d in R_SPARQL(q))
    
    Args:
        clip_similarity_matrix: CLIP similarity scores, shape (N_queries, N_artefacts)
        text2sparql_results: Dict mapping query UUID -> list of artefact URIs
        query_uuids: List of query UUIDs, length N_queries (rows of similarity matrix)
        artefact_uuids: List of artefact UUIDs, length N_artefacts (columns of similarity matrix)
        alpha: Weight for CLIP scores (default: 0.7)
        sparql_weight: Weight for SPARQL binary scores (default: 0.3)
                      Note: alpha + sparql_weight should = 1.0
    
    Returns:
        Fused similarity matrix, shape (N_queries, N_artefacts)
    """
    # Validate inputs
    assert clip_similarity_matrix.shape[0] == len(query_uuids), \
        f"Similarity matrix rows ({clip_similarity_matrix.shape[0]}) != query_uuids length ({len(query_uuids)})"
    assert clip_similarity_matrix.shape[1] == len(artefact_uuids), \
        f"Similarity matrix cols ({clip_similarity_matrix.shape[1]}) != artefact_uuids length ({len(artefact_uuids)})"
    
    # Ensure weights sum to 1
    if not np.isclose(alpha + sparql_weight, 1.0):
        print(f"Warning: alpha ({alpha}) + sparql_weight ({sparql_weight}) != 1.0, normalizing...")
        total = alpha + sparql_weight
        alpha = alpha / total
        sparql_weight = sparql_weight / total
    
    # Create artefact UUID to index mapping
    artefact_to_idx = {uuid: idx for idx, uuid in enumerate(artefact_uuids)}
    
    # Initialize SPARQL binary matrix (N_queries, N_artefacts)
    sparql_matrix = np.zeros_like(clip_similarity_matrix)
    
    # Fill SPARQL matrix
    for query_idx, query_uuid in enumerate(query_uuids):
        # Get SPARQL results for this query
        sparql_artefacts = text2sparql_results.get(query_uuid, [])
        
        # Set binary indicator for each artefact in SPARQL results
        for artefact_uri in sparql_artefacts:
            # Extract UUID from URI if needed
            # Assuming artefact_uri might be full URI or just UUID
            artefact_uuid = artefact_uri.split('/')[-1] if '/' in artefact_uri else artefact_uri
            
            if artefact_uuid in artefact_to_idx:
                artefact_idx = artefact_to_idx[artefact_uuid]
                sparql_matrix[query_idx, artefact_idx] = 1.0
    
    # Weighted fusion
    fused_matrix = alpha * clip_similarity_matrix + sparql_weight * sparql_matrix
    
    return fused_matrix


def additive_bonus_fusion(
    clip_similarity_matrix: np.ndarray,
    text2sparql_results: Dict[str, List[str]],
    query_uuids: List[str],
    artefact_uuids: List[str],
    delta: float = 0.5
) -> np.ndarray:
    """
    Fuse CLIP similarity scores with Text2SPARQL using additive bonus.
    
    Formula:
        S_final(q, d) = S_CLIP(q, d) + I(d in R_SPARQL(q)) * delta
    
    Args:
        clip_similarity_matrix: CLIP similarity scores, shape (N_queries, N_artefacts)
        text2sparql_results: Dict mapping query UUID -> list of artefact URIs
        query_uuids: List of query UUIDs, length N_queries
        artefact_uuids: List of artefact UUIDs, length N_artefacts
        delta: Bonus value for SPARQL matches (default: 0.5)
    
    Returns:
        Fused similarity matrix, shape (N_queries, N_artefacts)
    """
    # Validate inputs
    assert clip_similarity_matrix.shape[0] == len(query_uuids)
    assert clip_similarity_matrix.shape[1] == len(artefact_uuids)
    
    # Create artefact UUID to index mapping
    artefact_to_idx = {uuid: idx for idx, uuid in enumerate(artefact_uuids)}
    
    # Start with CLIP scores
    fused_matrix = clip_similarity_matrix.copy()
    
    # Add bonus for SPARQL matches
    for query_idx, query_uuid in enumerate(query_uuids):
        sparql_artefacts = text2sparql_results.get(query_uuid, [])
        
        for artefact_uri in sparql_artefacts:
            artefact_uuid = artefact_uri.split('/')[-1] if '/' in artefact_uri else artefact_uri
            
            if artefact_uuid in artefact_to_idx:
                artefact_idx = artefact_to_idx[artefact_uuid]
                fused_matrix[query_idx, artefact_idx] += delta
    
    return fused_matrix


def adaptive_additive_fusion(
    clip_similarity_matrix: np.ndarray,
    text2sparql_results: Dict[str, List[str]],
    query_uuids: List[str],
    artefact_uuids: List[str],
    delta: float = 0.5,
    size_thresholds: Dict[str, float] = None
) -> np.ndarray:
    """
    Fuse CLIP scores with Text2SPARQL using adaptive additive bonus based on result set size.
    
    Formula:
        S_final(q, d) = S_CLIP(q, d) + I(d in R_SPARQL(q)) * delta * omega(|R_SPARQL(q)|)
    
    where omega is a decay function based on SPARQL result set size.
    
    Args:
        clip_similarity_matrix: CLIP similarity scores, shape (N_queries, N_artefacts)
        text2sparql_results: Dict mapping query UUID -> list of artefact URIs
        query_uuids: List of query UUIDs
        artefact_uuids: List of artefact UUIDs
        delta: Base bonus value
        size_thresholds: Dict mapping size ranges to omega values
                        Default: {1: 1.0, 5: 0.8, 20: 0.5, 50: 0.3, inf: 0.1}
    
    Returns:
        Fused similarity matrix, shape (N_queries, N_artefacts)
    """
    if size_thresholds is None:
        size_thresholds = {
            1: 1.0,    # Exact match
            5: 0.8,    # High precision
            20: 0.5,   # Medium precision
            50: 0.3,   # Low precision
            float('inf'): 0.1  # Very low selectivity
        }
    
    # Validate inputs
    assert clip_similarity_matrix.shape[0] == len(query_uuids)
    assert clip_similarity_matrix.shape[1] == len(artefact_uuids)
    
    # Create artefact UUID to index mapping
    artefact_to_idx = {uuid: idx for idx, uuid in enumerate(artefact_uuids)}
    
    # Start with CLIP scores
    fused_matrix = clip_similarity_matrix.copy()
    
    # Add adaptive bonus
    for query_idx, query_uuid in enumerate(query_uuids):
        sparql_artefacts = text2sparql_results.get(query_uuid, [])
        sparql_size = len(sparql_artefacts)
        
        if sparql_size == 0:
            continue  # No bonus for empty results
        
        # Determine omega based on result set size
        omega = 0.0
        sorted_thresholds = sorted(size_thresholds.items())
        for threshold, weight in sorted_thresholds:
            if sparql_size <= threshold:
                omega = weight
                break
        
        # Add bonus to matching artefacts
        for artefact_uri in sparql_artefacts:
            artefact_uuid = artefact_uri.split('/')[-1] if '/' in artefact_uri else artefact_uri
            
            if artefact_uuid in artefact_to_idx:
                artefact_idx = artefact_to_idx[artefact_uuid]
                fused_matrix[query_idx, artefact_idx] += delta * omega
    
    return fused_matrix


def fuse_clip_and_text2sparql(
    clip_similarity_matrix: np.ndarray,
    text2sparql_results: Dict[str, List[str]],
    query_uuids: List[str],
    artefact_uuids: List[str],
    fusion_strategy: str = "weighted",
    fusion_params: Dict = None
) -> np.ndarray:
    """
    Complete fusion pipeline: CLIP (T2I + T2T) + Text2SPARQL.
    
    Args:
        t2i_similarity_matrix: Text-to-Image similarity, shape (N, N)
        t2t_similarity_matrix: Text-to-Text similarity, shape (N, N)
        text2sparql_json: Path to JSON file with Text2SPARQL results
        query_uuids: List of query UUIDs
        artefact_uuids: List of artefact UUIDs
        t2i_weight: Weight for T2I similarity (default: 0.1)
        t2t_weight: Weight for T2T similarity (default: 0.9)
        fusion_strategy: One of "weighted", "additive", "adaptive" (default: "weighted")
        fusion_params: Parameters for fusion strategy, e.g., {"alpha": 0.7, "sparql_weight": 0.3}
    
    Returns:
        Final fused similarity matrix
    """

    # Step 3: Fuse with Text2SPARQL
    if fusion_params is None:
        fusion_params = {}
    
    if fusion_strategy == "weighted":
        alpha = fusion_params.get("alpha", 0.7)
        sparql_weight = fusion_params.get("sparql_weight", 0.3)
        fused_matrix = weighted_fusion(
            clip_similarity_matrix,
            text2sparql_results,
            query_uuids,
            artefact_uuids,
            alpha=alpha,
            sparql_weight=sparql_weight
        )
    
    elif fusion_strategy == "additive":
        delta = fusion_params.get("delta", 0.5)
        fused_matrix = additive_bonus_fusion(
            clip_similarity_matrix,
            text2sparql_results,
            query_uuids,
            artefact_uuids,
            delta=delta
        )
    
    elif fusion_strategy == "adaptive":
        delta = fusion_params.get("delta", 0.5)
        size_thresholds = fusion_params.get("size_thresholds", None)
        fused_matrix = adaptive_additive_fusion(
            clip_similarity_matrix,
            text2sparql_results,
            query_uuids,
            artefact_uuids,
            delta=delta,
            size_thresholds=size_thresholds
        )
    
    else:
        raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    return fused_matrix