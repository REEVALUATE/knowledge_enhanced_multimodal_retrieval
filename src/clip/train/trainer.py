"""
CLIP training script with DDP support and joint T2I+T2T loss.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import random

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import clip

from ..utils.data_utils import load_splits_from_json
from ..utils.logging_utils import setup_logger, log_metrics_to_jsonl

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..model.clip_model import load_clip_model
from ..datasets.clip_dataset import CLIPTrainDataset, CLIPEvalDataset, collate_fn_train, collate_fn_eval
from .losses import JointContrastiveLoss
from ..eval.evaluator import evaluate_clip_model_for_training

logger = logging.getLogger(__name__)

# Environment settings
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.multiprocessing.set_sharing_strategy("file_system")


def setup_ddp(rank, world_size):
    """Initialize DDP environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


class CLIPTrainer:
    """
    CLIP trainer with joint T2I+T2T loss and DDP support.
    """
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        val_eval_dataset: Optional[CLIPEvalDataset],
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        output_dir: str = "./outputs",
        total_epochs: int = 20,
        early_stopping_patience: int = 5,
        early_stopping_metric: str = "avg",  # "avg", "t2i", "t2t"
        use_wandb: bool = False,
        wandb_project: str = "clip-finetuning",
        wandb_run_name: str = "experiment",
        grad_clip: float = 1.0,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_eval_dataset = val_eval_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.output_dir = Path(output_dir)
        self.total_epochs = total_epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Create output dir on rank 0
        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_file = self.output_dir / "metrics_epoch.jsonl"
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None
        
        # Early stopping (only on rank 0)
        self.best_metric = float('-inf')
        self.patience_counter = 0
        self.best_epoch = 0
        
        # WandB (only on rank 0)
        self.use_wandb = use_wandb and WANDB_AVAILABLE and self.rank == 0
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "total_epochs": total_epochs,
                    "batch_size": train_loader.batch_size * world_size,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "world_size": world_size,
                    "early_stopping_metric": early_stopping_metric,
                    "gradient_accumulation_steps": gradient_accumulation_steps
                }
            )
        
        if self.rank == 0:
            logger.info(f"Trainer initialized (Rank {rank}/{world_size})")
            logger.info(f"Early stopping: patience={early_stopping_patience}, metric={early_stopping_metric}")
            logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    def _get_model(self):
        """Get the underlying model (unwrap DDP if needed)."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.loss_fn.train()
        
        total_loss = 0.0
        total_loss_t2i = 0.0
        total_loss_t2t = 0.0
        num_batches = 0
        
        # Set epoch for distributed sampler
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, queries, targets, uuids) in enumerate(self.train_loader):
            images = images.to(self.device)
            
            # Tokenize texts
            query_tokens = clip.tokenize(queries, truncate=True).to(self.device)
            target_tokens = clip.tokenize(targets, truncate=True).to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.mixed_precision):

                base_model = self._get_model()
                
                image_features = base_model.encode_image(images)
                query_features = base_model.encode_text(query_tokens)
                target_features = base_model.encode_text(target_tokens)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                query_features = query_features / query_features.norm(dim=-1, keepdim=True)
                target_features = target_features / target_features.norm(dim=-1, keepdim=True)
                
                # Compute loss
                loss, metrics = self.loss_fn(image_features, query_features, target_features)
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Accumulate metrics
            total_loss += metrics['loss'] * self.gradient_accumulation_steps
            total_loss_t2i += metrics['loss_t2i']
            total_loss_t2t += metrics['loss_t2t']
            num_batches += 1
            
            # Log progress
            if self.rank == 0 and batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch+1} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {metrics['loss']:.4f} (T2I: {metrics['loss_t2i']:.4f}, T2T: {metrics['loss_t2t']:.4f})"
                )
        
        # Average metrics
        avg_metrics = {
            'train_loss': total_loss / num_batches,
            'train_loss_t2i': total_loss_t2i / num_batches,
            'train_loss_t2t': total_loss_t2t / num_batches
        }
        
        return avg_metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate on validation set."""
        if self.val_eval_dataset is None or self.rank != 0:
            return {}
        
        logger.info(f"Running validation...")
        
        # Use the evaluation function (MRR only for speed)
        base_model = self._get_model()
        
        metrics = evaluate_clip_model_for_training(
            model=base_model,
            dataset=self.val_eval_dataset,
            batch_size=64,
            device=self.device,
            seed=42,
            tasks=["T2I", "T2T"]  # Only evaluate T2I and T2T
        )
        
        # Compute average MRR for early stopping
        if self.early_stopping_metric == "avg":
            metrics['val_mrr_avg'] = (metrics['T2I_MRR'] + metrics['T2T_MRR']) / 2.0
        elif self.early_stopping_metric == "t2i":
            metrics['val_mrr_avg'] = metrics['T2I_MRR']
        elif self.early_stopping_metric == "t2t":
            metrics['val_mrr_avg'] = metrics['T2T_MRR']
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if self.rank != 0:
            return
        
        # Get base model (unwrap DDP)
        base_model = self._get_model()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': base_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest
        latest_path = self.output_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        logger.info(f"Saved checkpoint to {latest_path}")
        
        # Save best
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.total_epochs):
            if self.rank == 0:
                logger.info(f"\n{'='*50}")
                logger.info(f"Epoch {epoch+1}/{self.total_epochs}")
                logger.info(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Learning rate step
            if self.scheduler:
                self.scheduler.step()
            
            # Validate (only on rank 0)
            val_metrics = {}
            if self.rank == 0:
                val_metrics = self.validate(epoch)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch'] = epoch
            epoch_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            
            # Log to file and wandb (rank 0 only)
            if self.rank == 0:
                log_metrics_to_jsonl(epoch_metrics, self.metrics_file)
                
                if self.use_wandb:
                    wandb.log(epoch_metrics, step=epoch)
                
                # Print summary
                logger.info(f"Epoch {epoch+1} Summary:")
                logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
                if val_metrics:
                    logger.info(f"  Val T2I MRR: {val_metrics.get('T2I_MRR', 0):.2f}%")
                    logger.info(f"  Val T2T MRR: {val_metrics.get('T2T_MRR', 0):.2f}%")
                    logger.info(f"  Val Avg MRR: {val_metrics.get('val_mrr_avg', 0):.2f}%")
                
                # Early stopping check
                current_metric = val_metrics.get('val_mrr_avg', float('-inf'))
                is_best = current_metric > self.best_metric
                
                if is_best:
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    logger.info(f"  ✓ New best model! (MRR: {self.best_metric:.2f}%)")
                else:
                    self.patience_counter += 1
                    logger.info(f"  No improvement ({self.patience_counter}/{self.early_stopping_patience})")
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best=is_best)
                
                # Check early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"\nEarly stopping triggered! Best epoch: {self.best_epoch}")
                    break
        
        if self.rank == 0:
            logger.info(f"\nTraining complete! Best MRR: {self.best_metric:.2f}% (Epoch {self.best_epoch})")
            if self.use_wandb:
                wandb.finish()


def main_worker(rank, world_size, args):
    """Main worker function for DDP."""
    # Setup DDP
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    device = f'cuda:{rank}'
    
    # Setup logging (only detailed logging on rank 0)
    if rank == 0:
        log_file = Path(args.output_dir) / args.experiment_name / "train.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        setup_logger(__name__, str(log_file))
        logger.info(f"Training on {world_size} GPU(s)")
    
    # Set random seeds
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    
    # Load data splits
    logger.info(f"Loading data splits from {args.splits_file}") if rank == 0 else None
    train_uuids, val_uuids, test_uuids = load_splits_from_json(args.splits_file)
    
    if rank == 0:
        logger.info(f"Train: {len(train_uuids)}, Val: {len(val_uuids)}, Test: {len(test_uuids)}")
    
    # Load model
    if rank == 0:
        logger.info(f"Loading model: {args.model_name}")
    
    model, preprocess = load_clip_model(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        device=device
    )

    model = model.float()
    
    # Create datasets
    train_dataset = CLIPTrainDataset(
        uuids=train_uuids,
        image_folder=args.images_dir,
        text_folder=args.texts_dir,
        preprocessor=preprocess,
        max_text_length=args.max_text_length
    )
    
    val_dataset = CLIPTrainDataset(
        uuids=val_uuids,
        image_folder=args.images_dir,
        text_folder=args.texts_dir,
        preprocessor=preprocess,
        max_text_length=args.max_text_length
    )
    
    # Create evaluation dataset (only on rank 0)
    val_eval_dataset = None
    if rank == 0:
        val_eval_dataset = CLIPEvalDataset(
            uuids=val_uuids,
            image_folder=args.images_dir,
            text_folder=args.texts_dir,
            preprocessor=preprocess,
            max_text_length=args.max_text_length
        )
    
    # Create samplers
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_train,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_train
    )
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[rank],
            find_unused_parameters=False  # Set to False for efficiency
        )
    
    # Create loss function
    loss_fn = JointContrastiveLoss(
        temperature=args.temperature,
        t2i_weight=args.t2i_weight,
        t2t_weight=args.t2t_weight
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),  # CLIP default betas
        eps=1e-6
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr * 0.1
    )
    
    # Create trainer
    output_dir = Path(args.output_dir) / args.experiment_name
    
    trainer = CLIPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_eval_dataset=val_eval_dataset,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        rank=rank,
        world_size=world_size,
        output_dir=str(output_dir),
        total_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.experiment_name,
        grad_clip=args.grad_clip,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Train
    trainer.train()
    
    # Cleanup
    if world_size > 1:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='Fine-tune CLIP with joint T2I+T2T loss')
    
    # Model
    parser.add_argument('--model_name', type=str, default='ViT-L/14',
                       choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'])
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Data
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--texts_dir', type=str, required=True,
                       help='Directory with query-target JSON files')
    parser.add_argument('--splits_file', type=str, required=True)
    parser.add_argument('--max_text_length', type=int, default=150,
                       help='Maximum text length in words')
    
    # Loss weights
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--t2i_weight', type=float, default=0.5,
                       help='Weight for T2I loss')
    parser.add_argument('--t2t_weight', type=float, default=0.5,
                       help='Weight for T2T loss')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of gradient accumulation steps')
    
    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--early_stopping_metric', type=str, default='avg',
                       choices=['avg', 't2i', 't2t'],
                       help='Metric for early stopping: avg=(T2I_MRR+T2T_MRR)/2, t2i=T2I_MRR, t2t=T2T_MRR')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (recommended)')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    
    # WandB
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='clip-art-retrieval')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    
    if world_size == 0:
        raise RuntimeError("No CUDA devices available!")
    
    if world_size > 1:
        print(f"Using DDP with {world_size} GPUs")
        torch.multiprocessing.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        print("Using single GPU")
        main_worker(0, 1, args)


if __name__ == "__main__":
    main()