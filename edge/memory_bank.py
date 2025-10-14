"""
Memory bank with Distribution-Aware Sampling (DAS) for edge devices.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from collections import deque


class DistributionAwareSampling:
    """Gaussian Mixture Model for distribution-aware negative sampling."""
    
    def __init__(self, num_components: int, embedding_dim: int, 
                 alpha: float = 1.0, epsilon: float = 1e-6):
        """
        Initialize DAS with GMM.
        
        Args:
            num_components: Number of Gaussian components
            embedding_dim: Dimension of embeddings
            alpha: Sampling temperature
            epsilon: Small constant for numerical stability
        """
        self.num_components = num_components
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.epsilon = epsilon
        
        # GMM parameters
        self.pi = torch.ones(num_components) / num_components
        self.mu = torch.randn(num_components, embedding_dim) * 0.1
        self.sigma = torch.ones(num_components, embedding_dim) * 0.5
        
        # Statistics for online update
        self.n_samples = 0
        
    def pdf(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute probability density for embeddings.
        
        Args:
            embeddings: Tensor of shape [N, D]
            
        Returns:
            Probability densities [N]
        """
        N = embeddings.shape[0]
        densities = torch.zeros(N)
        
        for k in range(self.num_components):
            # Gaussian PDF
            diff = embeddings - self.mu[k]
            exponent = -0.5 * torch.sum((diff ** 2) / self.sigma[k], dim=1)
            normalizer = torch.prod(torch.sqrt(2 * np.pi * self.sigma[k]))
            densities += self.pi[k] * torch.exp(exponent) / normalizer
            
        return densities
    
    def get_sampling_probabilities(self, 
                                  embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute sampling probabilities (inverse of density).
        
        Args:
            embeddings: Tensor of shape [N, D]
            
        Returns:
            Sampling probabilities [N]
        """
        pdf_values = self.pdf(embeddings)
        # Lower density -> higher sampling probability
        probs = (pdf_values + self.epsilon) ** (-self.alpha)
        probs = probs / probs.sum()
        return probs
    
    def update(self, embeddings: torch.Tensor, lr: float = 0.01):
        """
        Online update of GMM parameters.
        
        Args:
            embeddings: New embeddings [N, D]
            lr: Learning rate for updates
        """
        with torch.no_grad():
            N = embeddings.shape[0]
            
            # E-step: Compute responsibilities
            responsibilities = torch.zeros(N, self.num_components)
            for k in range(self.num_components):
                diff = embeddings - self.mu[k]
                exponent = -0.5 * torch.sum((diff ** 2) / self.sigma[k], 
                                           dim=1)
                normalizer = torch.prod(torch.sqrt(2 * np.pi * self.sigma[k]))
                responsibilities[:, k] = (self.pi[k] * 
                                         torch.exp(exponent) / normalizer)
            
            responsibilities = (responsibilities / 
                               (responsibilities.sum(dim=1, keepdim=True) + 
                                self.epsilon))
            
            # M-step: Update parameters with momentum
            for k in range(self.num_components):
                r_k = responsibilities[:, k]
                N_k = r_k.sum()
                
                if N_k > 0:
                    # Update mixture weight
                    new_pi_k = N_k / N
                    self.pi[k] = (1 - lr) * self.pi[k] + lr * new_pi_k
                    
                    # Update mean
                    new_mu_k = (r_k.unsqueeze(1) * embeddings).sum(0) / N_k
                    self.mu[k] = (1 - lr) * self.mu[k] + lr * new_mu_k
                    
                    # Update variance
                    diff = embeddings - self.mu[k]
                    new_sigma_k = ((r_k.unsqueeze(1) * diff ** 2).sum(0) / 
                                  N_k)
                    self.sigma[k] = ((1 - lr) * self.sigma[k] + 
                                    lr * new_sigma_k)
                    
            # Normalize mixture weights
            self.pi = self.pi / self.pi.sum()
            self.n_samples += N


class MemoryBank:
    """Memory bank for storing negative samples with adaptive sizing."""
    
    def __init__(self, min_size: int, max_size: int, embedding_dim: int,
                 num_components: int = 5, alpha: float = 1.0):
        """
        Initialize memory bank.
        
        Args:
            min_size: Minimum bank size
            max_size: Maximum bank size
            embedding_dim: Dimension of embeddings
            num_components: Number of GMM components for DAS
            alpha: Sampling temperature
        """
        self.min_size = min_size
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        
        # Storage
        self.embeddings = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        
        # Distribution-aware sampling
        self.das = DistributionAwareSampling(num_components, embedding_dim, 
                                            alpha)
        
        # Current capacity
        self.current_capacity = max_size
        
    def push(self, embedding: torch.Tensor, timestamp: float):
        """
        Add embedding to memory bank.
        
        Args:
            embedding: Embedding vector [D]
            timestamp: Timestamp of embedding
        """
        self.embeddings.append(embedding.detach().cpu())
        self.timestamps.append(timestamp)
        
        # Update DAS periodically
        if len(self.embeddings) % 100 == 0 and len(self.embeddings) > 0:
            emb_tensor = torch.stack(list(self.embeddings))
            self.das.update(emb_tensor)
    
    def sample(self, n_samples: int, 
               use_das: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample negative examples from memory bank.
        
        Args:
            n_samples: Number of samples to draw
            use_das: Whether to use distribution-aware sampling
            
        Returns:
            Tuple of (embeddings, weights)
        """
        if len(self.embeddings) == 0:
            return None, None
            
        n_samples = min(n_samples, len(self.embeddings))
        emb_tensor = torch.stack(list(self.embeddings))
        
        if use_das and len(self.embeddings) >= self.min_size:
            # Distribution-aware sampling
            probs = self.das.get_sampling_probabilities(emb_tensor)
            indices = torch.multinomial(probs, n_samples, replacement=False)
        else:
            # Uniform random sampling
            indices = torch.randperm(len(self.embeddings))[:n_samples]
        
        sampled_embeddings = emb_tensor[indices]
        
        # Age-based weights
        current_time = self.timestamps[-1]
        sampled_timestamps = torch.tensor([self.timestamps[i] 
                                           for i in indices])
        ages = current_time - sampled_timestamps
        weights = torch.exp(-0.01 * ages)  # Exponential decay
        
        return sampled_embeddings, weights
    
    def update_capacity(self, available_memory_mb: float):
        """
        Adjust memory bank capacity based on available memory.
        
        Args:
            available_memory_mb: Available memory in MB
        """
        # Estimate memory per embedding (float32)
        bytes_per_embedding = self.embedding_dim * 4
        mb_per_embedding = bytes_per_embedding / (1024 * 1024)
        
        # Calculate capacity
        max_affordable = int(available_memory_mb / mb_per_embedding)
        self.current_capacity = max(self.min_size, 
                                   min(max_affordable, self.max_size))
        
        # Trim if necessary
        while len(self.embeddings) > self.current_capacity:
            self.embeddings.popleft()
            self.timestamps.popleft()
    
    def __len__(self):
        """Return current size of memory bank."""
        return len(self.embeddings)
