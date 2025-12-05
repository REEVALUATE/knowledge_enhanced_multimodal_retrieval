"""
Loss functions for CLIP fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SymmetricInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss (image-to-text + text-to-image).
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            image_features: (B, D) normalized
            text_features: (B, D) normalized
            
        Returns:
            loss, metrics_dict
        """
        batch_size = image_features.shape[0]
        
        # Compute similarity matrix
        logits = image_features @ text_features.T / self.temperature  # (B, B)
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Image-to-text loss
        loss_i2t = F.cross_entropy(logits, labels)
        
        # Text-to-image loss
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        # Symmetric loss
        loss = (loss_i2t + loss_t2i) / 2
        
        metrics = {
            'loss': loss.item(),
            'loss_i2t': loss_i2t.item(),
            'loss_t2i': loss_t2i.item()
        }
        
        return loss, metrics


class SymmetricInfoNCEWithT2TLoss(nn.Module):
    """
    Symmetric InfoNCE loss + Text-to-Text contrastive loss.
    FIXED: T2T loss now uses proper contrastive learning with negatives.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        t2t_weight: float = 0.5,
        i2t_weight: float = 0.5,
        t2i_weight: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.t2t_weight = t2t_weight
        self.i2t_weight = i2t_weight
        self.t2i_weight = t2i_weight
    def forward(
        self,
        image_features: torch.Tensor,
        text_features_v0: torch.Tensor,
        text_features_other: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            image_features: (B, D) normalized
            text_features_v0: (B, D) variant 0, normalized
            text_features_other: (B, 4, D) other 4 variants, normalized (optional)
            
        Returns:
            loss, metrics_dict
        """
        batch_size = image_features.shape[0]
        
        # 1. Symmetric InfoNCE loss (image <-> text variant 0)
        logits_i2t = image_features @ text_features_v0.T / self.temperature
        labels = torch.arange(batch_size, device=image_features.device)
        
        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss_t2i = F.cross_entropy(logits_i2t.T, labels)
        # loss_symmetric = (loss_i2t + loss_t2i) / 2
        w_sum = self.i2t_weight + self.t2i_weight
        loss_symmetric = (self.i2t_weight * loss_i2t + self.t2i_weight * loss_t2i) / w_sum
        
        metrics = {
            'loss_i2t': loss_i2t.item(),
            'loss_t2i': loss_t2i.item(),
            'loss_symmetric': loss_symmetric.item()
        }
        
        # 2. Text-to-text contrastive loss (if other variants provided)
        loss_t2t = torch.tensor(0.0, device=image_features.device)

        if text_features_other is not None:
            # text_features_other: (B, K, D)
            K = text_features_other.shape[1]
            text_other_flat = text_features_other.reshape(-1, text_features_other.shape[-1])  # (B*K, D)

            # Compute similarity matrix: (B, B*K)
            logits_t2t = text_features_v0 @ text_other_flat.T / self.temperature

            # Multi-positive soft cross-entropy
            losses_t2t = []
            for i in range(batch_size):
                query_logits = logits_t2t[i]  # (B*K,)

                targets = torch.zeros(batch_size * K, device=query_logits.device)
                targets[i * K:(i + 1) * K] = 1.0 / K

                log_probs = F.log_softmax(query_logits, dim=0)
                loss_i = -torch.sum(targets * log_probs)
                losses_t2t.append(loss_i)

            loss_t2t = torch.stack(losses_t2t).mean()
            metrics['loss_t2t'] = loss_t2t.item()
        
        # Total loss
        total_loss = loss_symmetric + self.t2t_weight * loss_t2t
        metrics['loss'] = total_loss.item()
        
        return total_loss, metrics


class MultiPositiveInfoNCELoss(nn.Module):
    """
    Multi-positive InfoNCE loss for training with 5 text variants.
    Each image has 5 positive text samples.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features_all: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            image_features: (B, D) normalized
            text_features_all: (B, 5, D) all 5 variants, normalized
            
        Returns:
            loss, metrics_dict
        """
        batch_size = image_features.shape[0]
        num_variants = text_features_all.shape[1]
        
        # Reshape text features to (B*5, D)
        text_features_flat = text_features_all.reshape(-1, text_features_all.shape[-1])
        
        # Compute similarity: (B, B*5)
        logits = image_features @ text_features_flat.T / self.temperature
        
        # Create labels: for image i, texts [i*5, i*5+1, ..., i*5+4] are all positive
        # We'll use multi-label cross-entropy
        
        # Method 1: Average over positive pairs
        losses = []
        for i in range(batch_size):
            # Get logits for image i
            img_logits = logits[i]  # (B*5,)
            
            # Positive indices
            pos_indices = list(range(i * num_variants, (i + 1) * num_variants))
            
            # Create targets
            targets = torch.zeros(batch_size * num_variants, device=logits.device)
            targets[pos_indices] = 1.0 / num_variants
            
            # Cross-entropy
            log_probs = F.log_softmax(img_logits, dim=0)
            loss_i = -torch.sum(targets * log_probs)
            losses.append(loss_i)
        
        loss_i2t = torch.stack(losses).mean()
        
        # Text-to-image loss (each text should find its image)
        losses_t2i = []
        for i in range(batch_size * num_variants):
            text_logits = logits[:, i]  # (B,)
            img_idx = i // num_variants
            loss_t = F.cross_entropy(text_logits.unsqueeze(0), torch.tensor([img_idx], device=logits.device))
            losses_t2i.append(loss_t)
        
        loss_t2i = torch.stack(losses_t2i).mean()
        
        # Symmetric loss
        loss = (loss_i2t + loss_t2i) / 2
        
        metrics = {
            'loss': loss.item(),
            'loss_i2t': loss_i2t.item(),
            'loss_t2i': loss_t2i.item()
        }
        
        return loss, metrics
