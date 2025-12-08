"""
Loss functions for CLIP fine-tuning with query-target pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class InfoNCELoss(nn.Module):
    """
    Standard symmetric InfoNCE loss (bidirectional contrastive learning).
    Used for both T2I and T2T tasks.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute symmetric InfoNCE loss.
        
        Args:
            features_a: (B, D) normalized features
            features_b: (B, D) normalized features
            
        Returns:
            loss: scalar tensor
            metrics: dict with loss components
        """
        batch_size = features_a.shape[0]
        device = features_a.device
        
        # Compute similarity matrix: (B, B)
        logits = (features_a @ features_b.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=device)
        
        # Bidirectional loss
        loss_a2b = F.cross_entropy(logits, labels)
        loss_b2a = F.cross_entropy(logits.T, labels)
        
        # Symmetric loss
        loss = (loss_a2b + loss_b2a) / 2.0
        
        metrics = {
            'loss': loss.item(),
            'loss_a2b': loss_a2b.item(),
            'loss_b2a': loss_b2a.item()
        }
        
        return loss, metrics


class JointContrastiveLoss(nn.Module):
    """
    Joint T2I + T2T contrastive loss for CLIP fine-tuning.
    
    Combines:
    - T2I loss: target_text ↔ image
    - T2T loss: query ↔ target_text
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        t2i_weight: float = 0.5,
        t2t_weight: float = 0.5
    ):
        """
        Args:
            temperature: Temperature for InfoNCE
            t2i_weight: Weight for T2I loss (default: 0.5)
            t2t_weight: Weight for T2T loss (default: 0.5)
        """
        super().__init__()
        self.temperature = temperature
        self.t2i_weight = t2i_weight
        self.t2t_weight = t2t_weight
        
        # Create InfoNCE loss module
        self.infonce = InfoNCELoss(temperature=temperature)
        
        # Normalize weights
        weight_sum = t2i_weight + t2t_weight
        self.t2i_weight = t2i_weight / weight_sum
        self.t2t_weight = t2t_weight / weight_sum
    
    def forward(
        self,
        image_features: torch.Tensor,
        query_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute joint contrastive loss.
        
        Args:
            image_features: (B, D) normalized image features
            query_features: (B, D) normalized query text features
            target_features: (B, D) normalized target text features
            
        Returns:
            total_loss: scalar tensor
            metrics: dict with all loss components
        """
        # T2I loss: target_text ↔ image
        loss_t2i, metrics_t2i = self.infonce(target_features, image_features)
        
        # T2T loss: query ↔ target_text
        loss_t2t, metrics_t2t = self.infonce(query_features, target_features)
        
        # Weighted combination
        total_loss = self.t2i_weight * loss_t2i + self.t2t_weight * loss_t2t
        
        # Combine metrics
        metrics = {
            'loss': total_loss.item(),
            'loss_t2i': loss_t2i.item(),
            'loss_t2t': loss_t2t.item(),
            't2i_weight': self.t2i_weight,
            't2t_weight': self.t2t_weight
        }
        
        return total_loss, metrics