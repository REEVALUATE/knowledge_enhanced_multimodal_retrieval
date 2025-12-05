"""
CLIP model wrapper with proper DDP support.
FIXED: Forces FP32 by ensuring visual.conv1 is float32
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
    device: str = 'cuda'
) -> Tuple[nn.Module, object]:
    """
    
    Args:
        model_name: CLIP model name (ViT-B/32, ViT-B/16, ViT-L/14)
        checkpoint_path: Path to checkpoint file (optional)
        device: Device to load model on
        
    Returns:
        model: CLIP model in FP32 
        preprocess: Image preprocessing function
    """
    logger.info(f"Loading CLIP model: {model_name}")

    clip_model, preprocess = clip.load(model_name, device=device, jit=False)

    clip_model = clip_model.float()

    if hasattr(clip_model, 'visual'):
        clip_model.visual = clip_model.visual.float()
        
        # Extra insurance: explicitly convert conv1
        if hasattr(clip_model.visual, 'conv1'):
            clip_model.visual.conv1 = clip_model.visual.conv1.float()
            logger.info("✓ Visual encoder conv1 forced to float32")
    
    # 3. Also ensure text encoder is float32
    if hasattr(clip_model, 'transformer'):
        clip_model.transformer = clip_model.transformer.float()
    
    if hasattr(clip_model, 'token_embedding'):
        clip_model.token_embedding = clip_model.token_embedding.float()
    
    if hasattr(clip_model, 'positional_embedding'):
        clip_model.positional_embedding = clip_model.positional_embedding.float()
    
    # 4. Verify all parameters are FP32
    param_dtype = next(clip_model.parameters()).dtype
    if param_dtype != torch.float32:
        logger.warning(f"⚠️ Parameters are {param_dtype}, forcing to float32")
        clip_model = clip_model.float()
    
    logger.info(f"✓ Model parameters: {param_dtype}")
    
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        clip_model.load_state_dict(state_dict, strict=True)
        
        # Re-apply FP32 after loading checkpoint
        clip_model = clip_model.float()
        if hasattr(clip_model, 'visual') and hasattr(clip_model.visual, 'conv1'):
            clip_model.visual.conv1 = clip_model.visual.conv1.float()
        
        logger.info(f"✓ Checkpoint loaded and converted to float32")
        
        # Log checkpoint info
        if 'epoch' in checkpoint:
            logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_metric' in checkpoint:
            logger.info(f"  Best metric: {checkpoint['best_metric']:.2f}")
    else:
        logger.info("Using pretrained CLIP from OpenAI")
    

    if device == 'cuda' and not clip_model.training:
        try:
            test_img = torch.randn(1, 3, 224, 224).to(device)
            clip_model.eval()
            with torch.no_grad():
                test_feat = clip_model.encode_image(test_img)
                output_dtype = test_feat.dtype
                
                if output_dtype != torch.float32:
                    logger.error(f"❌ Encode outputs {output_dtype} instead of float32!")
                    logger.error(f"   This will cause evaluation issues!")
                    
                    # Last resort: monkey patch encode methods
                    logger.info("   Applying emergency fix...")
                    clip_model = _force_fp32_monkey_patch(clip_model)
                    
                    # Test again
                    test_feat = clip_model.encode_image(test_img)
                    logger.info(f"   After fix: {test_feat.dtype}")
                else:
                    logger.info(f"✓ Encode output verified: {output_dtype}")
        except Exception as e:
            logger.warning(f"Could not verify encode dtype: {e}")
    
    # Final summary
    logger.info("="*60)
    logger.info("CLIP Model Loaded - FP32 Mode")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Dtype: {next(clip_model.parameters()).dtype}")
    logger.info("="*60)
    
    return clip_model, preprocess


def _force_fp32_monkey_patch(model):
    """
    Emergency fix: Monkey patch encode methods to remove dtype conversion.
    Only used if regular fix doesn't work.
    """
    logger.warning("Applying monkey patch to force FP32 encoding")
    
    # Save original methods
    original_encode_image = model.encode_image
    original_encode_text = model.encode_text
    
    # Create patched encode_image
    def patched_encode_image(image):
        # Force input to float32
        return model.visual(image.float())
    
    # Create patched encode_text  
    def patched_encode_text(text):
        x = model.token_embedding(text).float()
        x = x + model.positional_embedding.float()
        x = x.permute(1, 0, 2)
        x = model.transformer(x)
        x = x.permute(1, 0, 2)
        x = model.ln_final(x).float()
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection.float()
        return x
    
    # Replace methods
    model.encode_image = patched_encode_image
    model.encode_text = patched_encode_text
    
    logger.info("✓ Monkey patch applied")
    return model


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