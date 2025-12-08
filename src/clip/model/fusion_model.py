"""
Fusion head models for combining T2I and T2T retrieval scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGatedFusionWithBias(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.query_weight = nn.Parameter(torch.zeros(embed_dim))  # ✅ 从0开始
        self.bias = nn.Parameter(torch.tensor(-2.0))  # ✅ 初始化为-2

    def forward(self, query_embed, image_embed, target_embed):
        t2i_sim = query_embed @ image_embed.T
        t2t_sim = query_embed @ target_embed.T
        
        gate_logit = (query_embed * self.query_weight).sum(dim=1, keepdim=True) + self.bias
        gate = torch.sigmoid(gate_logit)  # ≈ 0.1初始
        
        scores = gate * t2i_sim + (1 - gate) * t2t_sim
        return scores

class LinearFusionHead(nn.Module):
    """Simple linear fusion of T2I and T2T scores."""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
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
        combined = torch.stack([t2i_sim, t2t_sim], dim=-1)  # (N, M, 2)
        scores = self.fusion(combined).squeeze(-1)  # (N, M)

        return scores


class CrossAttentionFusionHead(nn.Module):
    """Cross-attention fusion (query attends to T2I and T2T features)."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.image_proj = nn.Linear(embed_dim, embed_dim)
        self.target_proj = nn.Linear(embed_dim, embed_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
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
            query_embed: (N, D)
            image_embed: (M, D)
            target_embed: (M, D)
        Returns:
            scores: (N, M)
        """
        N = query_embed.shape[0]
        M = image_embed.shape[0]
        D = query_embed.shape[1]
        
        # Project
        query = self.query_proj(query_embed)
        image = self.image_proj(image_embed)
        target = self.target_proj(target_embed)
        
        # Expand for all pairs
        query_exp = query.unsqueeze(1).expand(N, M, D)
        image_exp = image.unsqueeze(0).expand(N, M, D)
        target_exp = target.unsqueeze(0).expand(N, M, D)
        
        # Reshape
        query_flat = query_exp.reshape(N * M, 1, D)
        kv_flat = torch.stack([
            image_exp.reshape(N * M, D),
            target_exp.reshape(N * M, D)
        ], dim=1)  # (N*M, 2, D)
        
        # Cross-attention
        attn_output, _ = self.cross_attn(
            query=query_flat,
            key=kv_flat,
            value=kv_flat
        )
        
        # Score
        attn_out = attn_output.squeeze(1)  # (N*M, D)
        scores_flat = self.score_mlp(attn_out).squeeze(-1)  # (N*M,)

        scores_flat = torch.tanh(scores_flat) * 0.5  # 输出范围 [-0.5, 0.5]
        
        scores = scores_flat.reshape(N, M)
        return scores


class GatedFusionHead(nn.Module):
    """
    Gated fusion: Learn query-specific weight for T2I vs T2T.
    Simple and effective!
    """
    
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        
        # Gate network: query -> weight for T2I
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(
        self,
        query_embed: torch.Tensor,
        image_embed: torch.Tensor,
        target_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_embed: (N, D)
            image_embed: (M, D)
            target_embed: (M, D)
        Returns:
            scores: (N, M)
        """
        # Compute base similarities
        t2i_sim = query_embed @ image_embed.T  # (N, M)
        t2t_sim = query_embed @ target_embed.T  # (N, M)
        
        # Compute query-specific gate (weight for T2I)
        gate = self.gate_net(query_embed)  # (N, 1)
        
        # Weighted fusion
        # gate = 1 → use T2I
        # gate = 0 → use T2T
        scores = gate * t2i_sim + (1 - gate) * t2t_sim  # (N, M)
        
        return scores

class SimpleGatedFusion(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.query_weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, query_embed, image_embed, target_embed):
        t2i_sim = query_embed @ image_embed.T
        t2t_sim = query_embed @ target_embed.T
        
        gate_logit = (query_embed * self.query_weight).sum(dim=1, keepdim=True) + self.bias
        gate = torch.sigmoid(gate_logit)
        
        scores = gate * t2i_sim + (1 - gate) * t2t_sim
        return scores
    
class BilinearFusionHead(nn.Module):
    """
    Bilinear fusion: Learn transformation for image and target separately.
    Then compute similarity with query.
    """
    
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        
        # Bilinear layers (no bias for symmetry)
        self.W_image = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_target = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Optional: learnable weight for combining
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Weight for T2I
    
    def forward(
        self,
        query_embed: torch.Tensor,
        image_embed: torch.Tensor,
        target_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_embed: (N, D)
            image_embed: (M, D)
            target_embed: (M, D)
        Returns:
            scores: (N, M)
        """
        # Project image and target
        image_proj = self.W_image(image_embed)  # (M, D)
        target_proj = self.W_target(target_embed)  # (M, D)
        
        # Bilinear scores
        t2i_scores = query_embed @ image_proj.T  # (N, M)
        t2t_scores = query_embed @ target_proj.T  # (N, M)
        
        # Weighted combination
        alpha = torch.sigmoid(self.alpha)  # Constrain to [0, 1]
        scores = alpha * t2i_scores + (1 - alpha) * t2t_scores
        
        return scores


class FusionModel(nn.Module):
    """Wrapper for fusion model with frozen CLIP encoder."""
    
    def __init__(
        self,
        clip_model,
        fusion_type: str = "linear",
        embed_dim: int = 768
    ):
        """
        Args:
            clip_model: Frozen CLIP model
            fusion_type: "linear", "cross_attention", "gated", or "bilinear"
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
        elif fusion_type == "gated":
            self.fusion_head = GatedFusionHead(embed_dim=embed_dim)
        elif fusion_type == "simple_gated":
            self.fusion_head = SimpleGatedFusion(embed_dim=embed_dim)
        elif fusion_type == "simple_gated_with_bias":
            self.fusion_head = SimpleGatedFusionWithBias(embed_dim=embed_dim)
        elif fusion_type == "bilinear":
            self.fusion_head = BilinearFusionHead(embed_dim=embed_dim)
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
            # Compute similarities first
            t2i_sim = query_embed @ image_embed.T  # (N, M)
            t2t_sim = query_embed @ target_embed.T  # (N, M)
            scores = self.fusion_head(t2i_sim, t2t_sim)
        
        else:
            # For other types, pass embeddings directly
            scores = self.fusion_head(query_embed, image_embed, target_embed)
        
        return scores