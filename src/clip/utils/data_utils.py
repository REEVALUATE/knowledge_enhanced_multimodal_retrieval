"""
Shared utility functions for data splitting and text selection.
Ensures deterministic behavior across all experiments.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def get_data_splits(
    images_dir: str,
    texts_dir: str,
    test_size: float = 0.15,
    val_size: float = 0.1,
    min_samples_for_split: int = 3,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split dataset based on object_type with stratification.
    Uses fixed random seed for reproducibility.
    
    Args:
        images_dir: Path to images directory
        texts_dir: Path to texts directory
        test_size: Test set proportion
        val_size: Validation set proportion (relative to train+val)
        min_samples_for_split: Minimum samples per class for stratification
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        train_uuids, val_uuids, test_uuids
    """
    from sklearn.model_selection import train_test_split
    
    texts_dir = Path(texts_dir)
    images_dir = Path(images_dir)
    
    # Get valid UUIDs
    text_uuids = set(f.stem for f in texts_dir.glob("*.json"))
    image_uuids = set()
    
    for ext in ['.jpg', '.jpeg', '.png']:
        image_uuids.update(f.stem for f in images_dir.glob(f"*{ext}"))
    
    valid_uuids = list(text_uuids & image_uuids)
    logger.info(f"Found {len(valid_uuids)} valid samples")
    
    # Read object_type for each sample
    uuid_to_type = {}
    type_counts = defaultdict(int)
    
    for uuid in valid_uuids:
        text_path = texts_dir / f"{uuid}.json"
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                object_type = data.get('object_type', 'Unknown')
                if not object_type or object_type.strip() == '':
                    object_type = 'Unknown'
                uuid_to_type[uuid] = object_type
                type_counts[object_type] += 1
        except:
            uuid_to_type[uuid] = 'Unknown'
            type_counts['Unknown'] += 1
    
    # Print statistics
    logger.info(f"\nObject Type distribution (top 10):")
    for obj_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {obj_type}: {count} ({count/len(valid_uuids)*100:.2f}%)")
    logger.info(f"Total types: {len(type_counts)}\n")
    
    # Separate small and large classes
    small_types = {t for t, c in type_counts.items() if c < min_samples_for_split}
    small_class_uuids = [uuid for uuid in valid_uuids if uuid_to_type[uuid] in small_types]
    large_class_uuids = [uuid for uuid in valid_uuids if uuid_to_type[uuid] not in small_types]
    
    logger.info(f"Small classes (<{min_samples_for_split} samples): {len(small_types)} types, {len(small_class_uuids)} samples → train")
    logger.info(f"Large classes (≥{min_samples_for_split} samples): {len(type_counts)-len(small_types)} types, {len(large_class_uuids)} samples → stratified split\n")
    
    # Stratified split for large classes
    large_class_labels = [uuid_to_type[uuid] for uuid in large_class_uuids]
    
    train_val_uuids, test_uuids = train_test_split(
        large_class_uuids,
        test_size=test_size,
        random_state=random_seed,
        stratify=large_class_labels
    )
    
    train_val_labels = [uuid_to_type[uuid] for uuid in train_val_uuids]
    train_uuids_large, val_uuids = train_test_split(
        train_val_uuids,
        test_size=val_size/(1-test_size),
        random_state=random_seed,
        stratify=train_val_labels
    )
    
    # Add small classes to train and shuffle
    train_uuids = train_uuids_large + small_class_uuids
    random.seed(random_seed)
    random.shuffle(train_uuids)
    
    logger.info(f"✓ Stratified split complete")
    logger.info(f"Train={len(train_uuids)} (large:{len(train_uuids_large)} + small:{len(small_class_uuids)})")
    logger.info(f"Val={len(val_uuids)}, Test={len(test_uuids)}\n")
    
    return train_uuids, val_uuids, test_uuids


def select_text_variant(
    uuid: str,
    epoch: int,
    num_variants: int = 5,
    random_seed: int = 42
) -> int:
    """
    Deterministically select a text variant for a given UUID and epoch.
    Same uuid + epoch will always return the same variant index.
    
    Args:
        uuid: Sample UUID
        epoch: Current epoch number
        num_variants: Total number of text variants (default: 5)
        random_seed: Base random seed
        
    Returns:
        Variant index (0 to num_variants-1)
    """
    # Create deterministic seed from uuid and epoch
    seed = hash((uuid, epoch, random_seed)) % (2**31)
    rng = random.Random(seed)
    return rng.randint(0, num_variants - 1)


def get_text_variant_for_batch(
    uuids: List[str],
    epoch: int,
    num_variants: int = 5,
    random_seed: int = 42
) -> List[int]:
    """
    Get text variant indices for a batch of UUIDs.
    
    Args:
        uuids: List of sample UUIDs
        epoch: Current epoch number
        num_variants: Total number of text variants
        random_seed: Base random seed
        
    Returns:
        List of variant indices
    """
    return [select_text_variant(uuid, epoch, num_variants, random_seed) for uuid in uuids]


def save_splits_to_json(
    train_uuids: List[str],
    val_uuids: List[str],
    test_uuids: List[str],
    output_path: str
):
    """Save data splits to JSON file for reproducibility."""
    splits = {
        'train': train_uuids,
        'val': val_uuids,
        'test': test_uuids,
        'train_size': len(train_uuids),
        'val_size': len(val_uuids),
        'test_size': len(test_uuids)
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
    
    logger.info(f"Splits saved to {output_path}")


def load_splits_from_json(input_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Load data splits from JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    logger.info(f"Loaded splits from {input_path}")
    logger.info(f"Train: {splits['train_size']}, Val: {splits['val_size']}, Test: {splits['test_size']}")
    
    return splits['train'], splits['val'], splits['test']
