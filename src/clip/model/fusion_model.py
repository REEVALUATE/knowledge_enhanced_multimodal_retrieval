"""
Fusion head models for combining T2I and T2T retrieval scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearFusionHead(nn.Module):
    """
    Simple linear fusion of T2I and T2T scores.
    Learns optimal weights for combining the two scores.
    """
    
    def __init__(self, hidden_dim: int = 128):
        """
        Args:
            hidden_dim: Hidden dimension for MLP (optional)
        """
        super().__init__()
        
        # Simple linear layer: [t2i_sim, t2t_sim] -> score
        self.fusion = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, t2i_sim: torch.Tensor, t2t_sim: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t2i_sim: (N, M) T2I similarity scores
            t2t_sim: (N, M) T2T similarity scores
            
        Returns:
            scores: (N, M) fused scores
        """
        # Stack similarities: (N, M, 2)
        combined = torch.stack([t2i_sim, t2t_sim], dim=-1)
        
        # Apply fusion: (N, M, 2) -> (N, M, 1) -> (N, M)
        scores = self.fusion(combined).squeeze(-1)
        
        return scores


class CrossAttentionFusionHead(nn.Module):
    """
    Cross-attention fusion that allows query to attend to T2I and T2T features.
    More sophisticated than linear fusion.
    """
    
    def __init__(
        self,
        embed_dim: int = 768,  # CLIP embedding dimension
        num_heads: int = 8,
        hidden_dim: int = 256
    ):
        """
        Args:
            embed_dim: Embedding dimension (768 for ViT-L/14)
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension for final MLP
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Project query, image, target embeddings to same space if needed
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.image_proj = nn.Linear(embed_dim, embed_dim)
        self.target_proj = nn.Linear(embed_dim, embed_dim)
        
        # Cross-attention: query attends to [image, target]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Final MLP to produce score
        self.score_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(
        self,
        query_embed: torch.Tensor,
        image_embed: torch.Tensor,
        target_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_embed: (N, D) query embeddings
            image_embed: (M, D) image embeddings (candidates)
            target_embed: (M, D) target text embeddings (candidates)
            
        Returns:
            scores: (N, M) fused scores
        """
        N = query_embed.shape[0]
        M = image_embed.shape[0]
        D = query_embed.shape[1]
        
        # Project embeddings
        query = self.query_proj(query_embed)  # (N, D)
        image = self.image_proj(image_embed)  # (M, D)
        target = self.target_proj(target_embed)  # (M, D)
        
        # Expand for all pairs: (N, M, D)
        query_exp = query.unsqueeze(1).expand(N, M, D)
        image_exp = image.unsqueeze(0).expand(N, M, D)
        target_exp = target.unsqueeze(0).expand(N, M, D)
        
        # Reshape to process all pairs in batch
        query_flat = query_exp.reshape(N * M, 1, D)  # (N*M, 1, D)
        
        # Stack image and target for each pair: (N*M, 2, D)
        kv_flat = torch.stack([
            image_exp.reshape(N * M, D),
            target_exp.reshape(N * M, D)
        ], dim=1)  # (N*M, 2, D)
        
        # Cross-attention: each query attends to its [image, target] pair
        attn_output, _ = self.cross_attn(
            query=query_flat,      # (N*M, 1, D)
            key=kv_flat,           # (N*M, 2, D)
            value=kv_flat          # (N*M, 2, D)
        )  # Output: (N*M, 1, D)
        
        # Score for each pair
        attn_out = attn_output.squeeze(1)  # (N*M, D)
        scores_flat = self.score_mlp(attn_out).squeeze(-1)  # (N*M,)
        
        # Reshape back to (N, M)
        scores = scores_flat.reshape(N, M)
        
        return scores


class FusionModel(nn.Module):
    """
    Wrapper for fusion model with frozen CLIP encoder.
    """
    
    def __init__(
        self,
        clip_model,
        fusion_type: str = "linear",  # "linear" or "cross_attention"
        embed_dim: int = 768
    ):
        """
        Args:
            clip_model: Frozen CLIP model
            fusion_type: Type of fusion head
            embed_dim: CLIP embedding dimension
        """
        super().__init__()
        
        self.clip_model = clip_model
        self.fusion_type = fusion_type
        
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Create fusion head
        if fusion_type == "linear":
            self.fusion_head = LinearFusionHead(hidden_dim=128)
        elif fusion_type == "cross_attention":
            self.fusion_head = CrossAttentionFusionHead(
                embed_dim=embed_dim,
                num_heads=8,
                hidden_dim=256
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    @torch.no_grad()
    def encode_query(self, query_tokens):
        """Encode query text."""
        features = self.clip_model.encode_text(query_tokens)
        return features / features.norm(dim=-1, keepdim=True)
    
    @torch.no_grad()
    def encode_target(self, target_tokens):
        """Encode target text."""
        features = self.clip_model.encode_text(target_tokens)
        return features / features.norm(dim=-1, keepdim=True)
    
    @torch.no_grad()
    def encode_image(self, images):
        """Encode images."""
        features = self.clip_model.encode_image(images)
        return features / features.norm(dim=-1, keepdim=True)
    
    def forward(
        self,
        query_embed: torch.Tensor,
        image_embed: torch.Tensor,
        target_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through fusion head.
        
        Args:
            query_embed: (N, D) normalized query embeddings
            image_embed: (M, D) normalized image embeddings
            target_embed: (M, D) normalized target embeddings
            
        Returns:
            scores: (N, M) fused scores
        """
        if self.fusion_type == "linear":
            # Compute similarities
            t2i_sim = query_embed @ image_embed.T  # (N, M)
            t2t_sim = query_embed @ target_embed.T  # (N, M)
            
            # Fuse
            scores = self.fusion_head(t2i_sim, t2t_sim)
        
        elif self.fusion_type == "cross_attention":
            # Use cross-attention fusion
            scores = self.fusion_head(query_embed, image_embed, target_embed)
        
        return scores