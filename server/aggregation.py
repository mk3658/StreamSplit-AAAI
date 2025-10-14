"""
Server-side aggregation and uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import time


class UncertaintyEstimator(nn.Module):
    """Estimate uncertainty for selective transmission."""
    
    def __init__(self, config: Dict, num_prototypes: int = 20):
        """
        Initialize uncertainty estimator.
        
        Args:
            config: Configuration dictionary
            num_prototypes: Number of prototype centers
        """
        super().__init__()
        
        self.config = config['server']['transmission']
        self.num_prototypes = num_prototypes
        
        # Prototype centers
        self.register_buffer('prototypes', 
                           torch.zeros(num_prototypes, 128))
        self.prototype_initialized = False
        
        # Weights for uncertainty components
        self.weights = self.config['weights']
        
    def consistency_uncertainty(self, embedding: torch.Tensor, 
                               embedding_aug: torch.Tensor) -> torch.Tensor:
        """Compute consistency-based uncertainty."""
        return torch.norm(embedding - embedding_aug, p=2, dim=-1)
    
    def entropy_uncertainty(self, embedding: torch.Tensor, 
                           classifier: Optional[nn.Module] = None) -> torch.Tensor:
        """Compute entropy-based uncertainty."""
        if classifier is None:
            # Use embedding norm as proxy
            return 1.0 / (torch.norm(embedding, dim=-1) + 1e-6)
        else:
            logits = classifier(embedding)
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            return entropy
    
    def prototype_uncertainty(self, embedding: torch.Tensor) -> torch.Tensor:
        """Compute prototype-based uncertainty."""
        if not self.prototype_initialized:
            return torch.ones(embedding.shape[0], device=embedding.device)
        
        # Distance to nearest prototype
        dists = torch.cdist(embedding, self.prototypes)
        min_dists, _ = torch.min(dists, dim=1)
        return min_dists
    
    def compute_uncertainty(self, 
                           embedding: torch.Tensor,
                           embedding_aug: Optional[torch.Tensor] = None,
                           classifier: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Compute combined uncertainty score.
        
        Args:
            embedding: Embedding tensor [batch, dim]
            embedding_aug: Augmented embedding [batch, dim]
            classifier: Optional classifier for entropy
            
        Returns:
            Uncertainty scores [batch]
        """
        # Consistency uncertainty
        if embedding_aug is not None:
            u_consistency = self.consistency_uncertainty(embedding, 
                                                        embedding_aug)
        else:
            u_consistency = torch.zeros(embedding.shape[0], 
                                       device=embedding.device)
        
        # Entropy uncertainty
        u_entropy = self.entropy_uncertainty(embedding, classifier)
        
        # Prototype uncertainty
        u_prototype = self.prototype_uncertainty(embedding)
        
        # Weighted combination
        uncertainty = (
            self.weights['consistency'] * u_consistency +
            self.weights['entropy'] * u_entropy +
            self.weights['prototype'] * u_prototype
        )
        
        return uncertainty
    
    def update_prototypes(self, embeddings: torch.Tensor, 
                         labels: Optional[torch.Tensor] = None):
        """Update prototype centers using k-means."""
        with torch.no_grad():
            if not self.prototype_initialized:
                # Initialize with k-means++
                indices = self._kmeans_plusplus_init(embeddings)
                self.prototypes.copy_(embeddings[indices])
                self.prototype_initialized = True
            else:
                # Update prototypes
                dists = torch.cdist(embeddings, self.prototypes)
                assignments = torch.argmin(dists, dim=1)
                
                for k in range(self.num_prototypes):
                    mask = assignments == k
                    if mask.sum() > 0:
                        self.prototypes[k] = embeddings[mask].mean(dim=0)
    
    def _kmeans_plusplus_init(self, embeddings: torch.Tensor) -> torch.Tensor:
        """K-means++ initialization."""
        n = embeddings.shape[0]
        indices = []
        
        # First center: random
        indices.append(torch.randint(0, n, (1,)).item())
        
        # Remaining centers
        for _ in range(1, self.num_prototypes):
            # Compute distances to nearest center
            centers = embeddings[indices]
            dists = torch.cdist(embeddings, centers)
            min_dists, _ = torch.min(dists, dim=1)
            
            # Sample proportional to distance squared
            probs = min_dists ** 2
            probs = probs / probs.sum()
            next_idx = torch.multinomial(probs, 1).item()
            indices.append(next_idx)
        
        return torch.tensor(indices, device=embeddings.device)


class ServerAggregator:
    """Aggregate embeddings from multiple edge devices."""
    
    def __init__(self, config: Dict):
        """Initialize aggregator."""
        self.config = config['server']
        self.device_embeddings = {}  # device_id -> list of (embedding, timestamp)
        self.kernel_bandwidth = (
            self.config['aggregation']['kernel_bandwidth']
        )
        
    def add_embedding(self, device_id: str, embedding: torch.Tensor, 
                     timestamp: float):
        """Add embedding from edge device."""
        if device_id not in self.device_embeddings:
            self.device_embeddings[device_id] = []
        
        self.device_embeddings[device_id].append((embedding, timestamp))
        
        # Keep only recent embeddings (last 1000)
        if len(self.device_embeddings[device_id]) > 1000:
            self.device_embeddings[device_id] = (
                self.device_embeddings[device_id][-1000:]
            )
    
    def aggregate_intra_device(self, 
                              device_id: str) -> Optional[torch.Tensor]:
        """
        Aggregate embeddings from single device using kernel density.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Aggregated distribution estimate or None
        """
        if device_id not in self.device_embeddings:
            return None
        
        embeddings_list = self.device_embeddings[device_id]
        if len(embeddings_list) == 0:
            return None
        
        # Extract embeddings
        embeddings = torch.stack([e[0] for e in embeddings_list])
        
        # Simple mean for now (can be enhanced with KDE)
        aggregated = torch.mean(embeddings, dim=0, keepdim=True)
        
        return aggregated
    
    def aggregate_cross_device(self) -> Optional[torch.Tensor]:
        """
        Aggregate across all devices.
        
        Returns:
            Global aggregated embeddings
        """
        device_aggregates = []
        
        for device_id in self.device_embeddings.keys():
            agg = self.aggregate_intra_device(device_id)
            if agg is not None:
                device_aggregates.append(agg)
        
        if len(device_aggregates) == 0:
            return None
        
        # Stack and average
        global_aggregate = torch.stack(device_aggregates).mean(dim=0)
        
        return global_aggregate
