"""
Streaming contrastive learning module for edge devices.
Implements local contrastive loss with momentum and consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import time


class StreamingContrastiveLearning(nn.Module):
    """Streaming contrastive learning on edge device."""
    
    def __init__(self, config: Dict, encoder: nn.Module, memory_bank):
        """
        Initialize streaming contrastive learning.
        
        Args:
            config: Configuration dictionary
            encoder: Neural network encoder
            memory_bank: MemoryBank instance
        """
        super().__init__()
        
        self.config = config['edge']
        self.encoder = encoder
        self.memory_bank = memory_bank
        
        # Create momentum encoder
        self.encoder_key = self._create_momentum_encoder()
        
        # Contrastive parameters
        self.temperature = self.config['contrastive']['temperature']
        self.momentum = self.config['contrastive']['momentum']
        self.lambda_consistency = (
            self.config['contrastive']['lambda_consistency']
        )
        
        # Gradient accumulation
        self.accumulation_steps = (
            self.config['contrastive']['gradient_accumulation']
        )
        self.accumulated_grads = 0
        
    def _create_momentum_encoder(self) -> nn.Module:
        """Create momentum encoder as copy of main encoder."""
        encoder_key = type(self.encoder)(
            **self.encoder.get_config() 
            if hasattr(self.encoder, 'get_config') else {}
        )
        encoder_key.load_state_dict(self.encoder.state_dict())
        
        # Freeze momentum encoder
        for param in encoder_key.parameters():
            param.requires_grad = False
            
        return encoder_key
    
    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Update momentum encoder with EMA."""
        for param_q, param_k in zip(self.encoder.parameters(), 
                                    self.encoder_key.parameters()):
            param_k.data = (self.momentum * param_k.data + 
                           (1 - self.momentum) * param_q.data)
    
    def forward(self, x: torch.Tensor, 
                x_aug: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for contrastive learning.
        
        Args:
            x: Anchor samples [batch_size, ...]
            x_aug: Augmented positive samples [batch_size, ...]
            
        Returns:
            Tuple of (loss, info_dict)
        """
        batch_size = x.shape[0]
        
        # Encode anchor with query encoder
        embeddings_q = self.encoder(x)
        embeddings_q = F.normalize(embeddings_q, dim=1)
        
        # Encode positive with key encoder (no gradient)
        with torch.no_grad():
            self._update_momentum_encoder()
            embeddings_k = self.encoder_key(x_aug)
            embeddings_k = F.normalize(embeddings_k, dim=1)
        
        # Compute local contrastive loss
        loss_contrastive = self._compute_contrastive_loss(
            embeddings_q, embeddings_k
        )
        
        # Compute consistency loss
        embeddings_q_aug = self.encoder(x_aug)
        embeddings_q_aug = F.normalize(embeddings_q_aug, dim=1)
        loss_consistency = F.mse_loss(embeddings_q_aug, embeddings_k)
        
        # Total loss
        loss = loss_contrastive + self.lambda_consistency * loss_consistency
        
        # Update memory bank
        current_time = time.time()
        for emb in embeddings_k:
            self.memory_bank.push(emb, current_time)
        
        # Info dict
        info = {
            'loss_contrastive': loss_contrastive.item(),
            'loss_consistency': loss_consistency.item(),
            'loss_total': loss.item(),
            'memory_bank_size': len(self.memory_bank)
        }
        
        return loss, info
    
    def _compute_contrastive_loss(self, embeddings_q: torch.Tensor, 
                                  embeddings_k: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss with memory bank negatives.
        
        Args:
            embeddings_q: Query embeddings [batch_size, dim]
            embeddings_k: Key (positive) embeddings [batch_size, dim]
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = embeddings_q.shape[0]
        
        # Positive logits
        pos_logits = torch.sum(embeddings_q * embeddings_k, dim=1, 
                              keepdim=True)
        pos_logits = pos_logits / self.temperature
        
        # Sample negatives from memory bank
        neg_embeddings, neg_weights = self.memory_bank.sample(
            n_samples=min(256, len(self.memory_bank))
        )
        
        if neg_embeddings is not None:
            neg_embeddings = neg_embeddings.to(embeddings_q.device)
            neg_weights = neg_weights.to(embeddings_q.device)
            
            # Negative logits
            neg_logits = torch.matmul(embeddings_q, 
                                     neg_embeddings.t())
            neg_logits = neg_logits / self.temperature
            
            # Apply age weights
            neg_logits = neg_logits * neg_weights.unsqueeze(0)
            
            # Concatenate positive and negative logits
            logits = torch.cat([pos_logits, neg_logits], dim=1)
        else:
            # No negatives available yet
            logits = pos_logits
        
        # Labels (positive is always first)
        labels = torch.zeros(batch_size, dtype=torch.long, 
                           device=embeddings_q.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def compute_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute embedding for input.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized embedding
        """
        with torch.no_grad():
            embedding = self.encoder(x)
            embedding = F.normalize(embedding, dim=1)
        return embedding


class PositivePairGenerator:
    """Generate positive pairs for contrastive learning."""
    
    def __init__(self, config: Dict):
        """Initialize pair generator."""
        self.config = config
        self.lambda_temporal = 0.1  # Temporal offset decay
        
    def generate_temporal_positive(self, 
                                   anchor_idx: int, 
                                   max_offset: int = 5) -> int:
        """
        Generate temporally close positive sample index.
        
        Args:
            anchor_idx: Index of anchor sample
            max_offset: Maximum temporal offset
            
        Returns:
            Index of positive sample
        """
        # Sample offset from exponential distribution
        offset = int(torch.distributions.Exponential(
            self.lambda_temporal
        ).sample().item())
        offset = min(offset, max_offset)
        
        # Random direction
        if torch.rand(1) < 0.5:
            offset = -offset
            
        positive_idx = anchor_idx + offset
        return positive_idx
