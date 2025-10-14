"""
Hybrid loss function combining Sliced-Wasserstein and Laplacian regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class SlicedWassersteinDistance(nn.Module):
    """Sliced-Wasserstein distance for distribution alignment."""
    
    def __init__(self, num_projections: int = 100):
        """
        Initialize Sliced-Wasserstein distance.
        
        Args:
            num_projections: Number of random projections
        """
        super().__init__()
        self.num_projections = num_projections
        
    def forward(self, embeddings_p: torch.Tensor, 
                embeddings_q: torch.Tensor) -> torch.Tensor:
        """
        Compute Sliced-Wasserstein distance between two sets of embeddings.
        
        Args:
            embeddings_p: First set [N, D]
            embeddings_q: Second set [M, D]
            
        Returns:
            SW distance scalar
        """
        device = embeddings_p.device
        dim = embeddings_p.shape[1]
        
        # Generate random projection directions
        projections = torch.randn(dim, self.num_projections, device=device)
        projections = F.normalize(projections, dim=0)
        
        # Project embeddings
        proj_p = torch.matmul(embeddings_p, projections)  # [N, L]
        proj_q = torch.matmul(embeddings_q, projections)  # [M, L]
        
        # Compute 1D Wasserstein distance for each projection
        sw_distance = 0.0
        for i in range(self.num_projections):
            # Sort projections
            sorted_p, _ = torch.sort(proj_p[:, i])
            sorted_q, _ = torch.sort(proj_q[:, i])
            
            # Interpolate to same size if needed
            if sorted_p.shape[0] != sorted_q.shape[0]:
                min_size = min(sorted_p.shape[0], sorted_q.shape[0])
                sorted_p = F.interpolate(
                    sorted_p.unsqueeze(0).unsqueeze(0),
                    size=min_size,
                    mode='linear',
                    align_corners=True
                ).squeeze()
                sorted_q = F.interpolate(
                    sorted_q.unsqueeze(0).unsqueeze(0),
                    size=min_size,
                    mode='linear',
                    align_corners=True
                ).squeeze()
            
            # 1D Wasserstein distance (L2)
            w_dist = torch.mean((sorted_p - sorted_q) ** 2)
            sw_distance += w_dist
        
        # Average over projections
        sw_distance = sw_distance / self.num_projections
        sw_distance = torch.sqrt(sw_distance + 1e-8)
        
        return sw_distance


class LaplacianRegularization(nn.Module):
    """Laplacian regularization for preserving local structure."""
    
    def __init__(self, k: int = 10, sigma: float = 0.5):
        """
        Initialize Laplacian regularization.
        
        Args:
            k: Number of nearest neighbors
            sigma: Bandwidth for Gaussian kernel
        """
        super().__init__()
        self.k = k
        self.sigma = sigma
        
    def _construct_graph(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Construct k-NN graph with Gaussian weights.
        
        Args:
            embeddings: Embedding tensor [N, D]
            
        Returns:
            Weight matrix [N, N]
        """
        N = embeddings.shape[0]
        
        # Compute pairwise distances
        dists = torch.cdist(embeddings, embeddings, p=2)
        
        # Find k nearest neighbors
        _, indices = torch.topk(dists, k=self.k + 1, largest=False, dim=1)
        
        # Construct weight matrix
        W = torch.zeros(N, N, device=embeddings.device)
        for i in range(N):
            for j in indices[i, 1:]:  # Skip self (first element)
                dist_ij = dists[i, j]
                weight = torch.exp(-dist_ij ** 2 / (2 * self.sigma ** 2))
                W[i, j] = weight
                W[j, i] = weight  # Symmetric
        
        return W
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian regularization loss.
        
        Args:
            embeddings: Embedding tensor [N, D]
            
        Returns:
            Regularization loss scalar
        """
        # Construct graph
        W = self._construct_graph(embeddings)
        
        # Degree matrix
        D = torch.diag(W.sum(dim=1))
        
        # Laplacian matrix
        L = D - W
        
        # Compute trace(E^T L E)
        loss = torch.trace(torch.matmul(
            torch.matmul(embeddings.t(), L), 
            embeddings
        ))
        
        # Normalize by number of samples
        loss = loss / embeddings.shape[0] ** 2
        
        return loss


class HybridLoss(nn.Module):
    """Hybrid loss combining Sliced-Wasserstein and Laplacian regularization."""
    
    def __init__(self, config: Dict):
        """
        Initialize hybrid loss.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        hybrid_config = config['server']['hybrid_loss']
        
        # Components
        self.sw_loss = SlicedWassersteinDistance(
            num_projections=hybrid_config['num_projections']
        )
        self.lap_loss = LaplacianRegularization(
            k=hybrid_config['k_nearest_neighbors'],
            sigma=hybrid_config['sigma_w']
        )
        
        # Weight for Laplacian term
        self.lambda_lap = hybrid_config['lambda_laplacian']
        
    def forward(self, 
                edge_embeddings: torch.Tensor,
                server_embeddings: Optional[torch.Tensor] = None,
                return_components: bool = False) -> torch.Tensor:
        """
        Compute hybrid loss.
        
        Args:
            edge_embeddings: Embeddings from edge devices [N, D]
            server_embeddings: Embeddings from server (optional) [M, D]
            return_components: Whether to return loss components
            
        Returns:
            Total loss or dict of components
        """
        # Sliced-Wasserstein component
        if server_embeddings is not None:
            loss_sw = self.sw_loss(edge_embeddings, server_embeddings)
        else:
            # Self-consistency
            loss_sw = torch.tensor(0.0, device=edge_embeddings.device)
        
        # Laplacian regularization component
        loss_lap = self.lap_loss(edge_embeddings)
        
        # Total loss
        loss_total = loss_sw + self.lambda_lap * loss_lap
        
        if return_components:
            return {
                'loss_total': loss_total,
                'loss_sw': loss_sw,
                'loss_lap': loss_lap
            }
        else:
            return loss_total
