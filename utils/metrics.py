"""
Evaluation metrics for StreamSplit.
"""

import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, silhouette_score)
from sklearn.manifold import TSNE
from typing import Dict, Tuple


def compute_accuracy(predictions: torch.Tensor, 
                    labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    pred_labels = torch.argmax(predictions, dim=1)
    return accuracy_score(labels.cpu().numpy(), pred_labels.cpu().numpy())


def compute_precision_recall_f1(predictions: torch.Tensor,
                                labels: torch.Tensor,
                                average: str = 'macro') -> Dict[str, float]:
    """Compute precision, recall, and F1 score."""
    pred_labels = torch.argmax(predictions, dim=1)
    y_true = labels.cpu().numpy()
    y_pred = pred_labels.cpu().numpy()
    
    return {
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1_score': f1_score(y_true, y_pred, average=average)
    }


def compute_retrieval_precision_at_k(embeddings: torch.Tensor,
                                    labels: torch.Tensor,
                                    k: int = 10) -> float:
    """Compute retrieval precision@k."""
    n = embeddings.shape[0]
    
    # Compute pairwise distances
    dists = torch.cdist(embeddings, embeddings)
    
    # Get k nearest neighbors (excluding self)
    _, indices = torch.topk(dists, k=k+1, largest=False, dim=1)
    indices = indices[:, 1:]  # Remove self
    
    # Compute precision
    correct = 0
    for i in range(n):
        query_label = labels[i]
        retrieved_labels = labels[indices[i]]
        correct += (retrieved_labels == query_label).sum().item()
    
    precision = correct / (n * k)
    return precision


def compute_silhouette_score(embeddings: torch.Tensor,
                             labels: torch.Tensor) -> float:
    """Compute silhouette score for clustering quality."""
    emb_np = embeddings.cpu().numpy()
    lab_np = labels.cpu().numpy()
    
    if len(np.unique(lab_np)) < 2:
        return 0.0
    
    return silhouette_score(emb_np, lab_np)


def compute_tsne_embeddings(embeddings: torch.Tensor,
                           perplexity: int = 30,
                           n_iter: int = 1000) -> np.ndarray:
    """Compute t-SNE embeddings for visualization."""
    emb_np = embeddings.cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=perplexity, 
               n_iter=n_iter, random_state=42)
    return tsne.fit_transform(emb_np)


class MetricsTracker:
    """Track and aggregate metrics during training."""
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
        
    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
    
    def compute_averages(self) -> Dict[str, float]:
        """Compute average of all metrics."""
        averages = {}
        for key in self.metrics:
            if self.counts[key] > 0:
                averages[key] = self.metrics[key] / self.counts[key]
            else:
                averages[key] = 0.0
        return averages
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on dataset.
    
    Args:
        model: Neural network model
        dataloader: DataLoader for evaluation
        device: Device to run on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            embeddings = model(inputs)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all batches
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    metrics = {
        'retrieval_precision@10': compute_retrieval_precision_at_k(
            all_embeddings, all_labels, k=10
        ),
        'silhouette_score': compute_silhouette_score(
            all_embeddings, all_labels
        )
    }
    
    return metrics
