"""
CLIP model wrapper with proper DDP support.
"""

import torch
import torch.nn as nn
import clip
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def load_clip_model(
    model_name: str = 'ViT-L/14',
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda:0'
) -> Tuple[nn.Module, object]:
    """
    Load CLIP model.
    
    Args:
        model_name: CLIP model name (ViT-B/32, ViT-B/16, ViT-L/14)
        checkpoint_path: Path to checkpoint file (optional)
        device: Device to load model on
        freeze_encoders: Deprecated, kept for compatibility
        
    Returns:
        model: CLIP model (not wrapped, ready for DDP in trainer)
        preprocess: Image preprocessing function
        
    Note:
        - Returns the raw CLIP model, NOT wrapped
        - DDP wrapping happens in the trainer
        - Model is always in float32 for stable training
    """
    logger.info(f"Loading CLIP model: {model_name}")
    
    # Load pretrained CLIP
    clip_model, preprocess = clip.load(model_name, device=device)
    
    # CRITICAL: Ensure float32 (CLIP sometimes loads in float16)
    clip_model = clip_model.float()
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  Loaded 'model_state_dict' from checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"  Loaded 'state_dict' from checkpoint")
        else:
            # Assume checkpoint is the state dict itself
            state_dict = checkpoint
            print(f"  Loaded checkpoint as state_dict directly")
        
        # Load state dict
        clip_model.load_state_dict(state_dict, strict=True)
        logger.info(f"Checkpoint loaded successfully")

        # Log checkpoint info if available
        if 'epoch' in checkpoint:
            logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_metric' in checkpoint:
            logger.info(f"  Best metric: {checkpoint['best_metric']:.2f}")
    else:
        logger.info("Using pretrained CLIP from OpenAI")
    
    return clip_model, preprocess


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    save_path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save (can be DDP-wrapped or not)
        optimizer: Optimizer
        epoch: Current epoch
        best_metric: Best validation metric
        best_epoch: Epoch with best metric
        save_path: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        
    Note:
        Automatically unwraps DDP if needed
    """
    # Unwrap DDP if needed
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_to_save = model.module
    else:
        model_to_save = model
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'best_epoch': best_epoch
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint_for_resuming(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cuda'
) -> Tuple[int, float, int]:
    """
    Load checkpoint to resume training.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load into (can be DDP-wrapped or not)
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into (optional)
        device: Device
        
    Returns:
        epoch: Epoch to resume from
        best_metric: Best metric so far
        best_epoch: Epoch with best metric
    """
    logger.info(f"Loading checkpoint for resuming: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Unwrap DDP if needed
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_to_load = model.module
    else:
        model_to_load = model
    
    # Load model state
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if available
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', float('-inf'))
    best_epoch = checkpoint.get('best_epoch', 0)
    
    logger.info(f"Resumed from epoch {epoch}, best metric: {best_metric:.2f}")
    
    return epoch, best_metric, best_epoch


def freeze_clip_encoders(model: nn.Module):
    """
    Freeze CLIP encoders (keep only projection layers trainable).
    
    Args:
        model: CLIP model (can be DDP-wrapped or not)
        
    Note:
        This is rarely needed in our setup since we train end-to-end
    """
    # Unwrap DDP if needed
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        clip_model = model.module
    else:
        clip_model = model
    
    logger.info("Freezing CLIP encoders...")
    
    # Freeze visual encoder except projection
    for name, param in clip_model.visual.named_parameters():
        if 'proj' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Freeze text encoder except projection and final layer norm
    for param in clip_model.transformer.parameters():
        param.requires_grad = False
    
    for param in clip_model.token_embedding.parameters():
        param.requires_grad = False
    
    if hasattr(clip_model, 'positional_embedding'):
        clip_model.positional_embedding.requires_grad = False
    
    # Keep projection trainable
    if hasattr(clip_model, 'text_projection') and clip_model.text_projection is not None:
        clip_model.text_projection.requires_grad = True
    
    # Keep final layer norm trainable
    if hasattr(clip_model, 'ln_final'):
        for param in clip_model.ln_final.parameters():
            param.requires_grad = True
    
    # Log trainable parameters
    total_params = sum(p.numel() for p in clip_model.parameters())
    trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")


def unfreeze_clip_encoders(model: nn.Module):
    """
    Unfreeze all CLIP parameters.
    
    Args:
        model: CLIP model (can be DDP-wrapped or not)
    """
    # Unwrap DDP if needed
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        clip_model = model.module
    else:
        clip_model = model
    
    logger.info("Unfreezing all CLIP parameters...")
    
    for param in clip_model.parameters():
        param.requires_grad = True
    
    total_params = sum(p.numel() for p in clip_model.parameters())
    logger.info(f"All {total_params:,} parameters are now trainable")


def get_trainable_params(model: nn.Module) -> int:
    """
    Get number of trainable parameters.
    
    Args:
        model: Model (can be DDP-wrapped or not)
        
    Returns:
        Number of trainable parameters
    """
    # Unwrap DDP if needed
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        actual_model = model.module
    else:
        actual_model = model
    
    return sum(p.numel() for p in actual_model.parameters() if p.requires_grad)


def print_model_info(model: nn.Module):
    """
    Print model information.
    
    Args:
        model: Model (can be DDP-wrapped or not)
    """
    # Unwrap DDP if needed
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        actual_model = model.module
        is_ddp = True
    else:
        actual_model = model
        is_ddp = False
    
    total_params = sum(p.numel() for p in actual_model.parameters())
    trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    
    logger.info("="*60)
    logger.info("Model Information")
    logger.info("="*60)
    logger.info(f"DDP wrapped: {is_ddp}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable percentage: {100*trainable_params/total_params:.2f}%")
    logger.info("="*60)