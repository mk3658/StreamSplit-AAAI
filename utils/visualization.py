"""
Visualization utilities for StreamSplit.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.manifold import TSNE
from typing import Optional


def plot_tsne(embeddings: torch.Tensor, labels: torch.Tensor,
              save_path: Optional[str] = None, title: str = "t-SNE"):
    """
    Plot t-SNE visualization of embeddings.
    
    Args:
        embeddings: Embedding tensor [N, D]
        labels: Label tensor [N]
        save_path: Path to save figure
        title: Plot title
    """
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings.cpu().numpy())
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                         c=labels.cpu().numpy(), 
                         cmap='tab10', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Class')
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(metrics_history: list, 
                         save_path: Optional[str] = None):
    """
    Plot training curves from metrics history.
    
    Args:
        metrics_history: List of metric dictionaries per epoch
        save_path: Path to save figure
    """
    epochs = [m['epoch'] for m in metrics_history]
    
    # Extract metrics
    metrics_to_plot = {}
    for metric_dict in metrics_history:
        for key, value in metric_dict['metrics'].items():
            if key not in metrics_to_plot:
                metrics_to_plot[key] = []
            metrics_to_plot[key].append(value)
    
    # Plot
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, (metric_name, values) in zip(axes, metrics_to_plot.items()):
        ax.plot(epochs, values, marker='o', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_resource_usage(timestamps: list, cpu_usage: list, 
                       memory_usage: list, 
                       save_path: Optional[str] = None):
    """
    Plot resource usage over time.
    
    Args:
        timestamps: List of timestamps
        cpu_usage: List of CPU usage percentages
        memory_usage: List of memory usage percentages
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # CPU usage
    ax1.plot(timestamps, cpu_usage, color='blue', linewidth=1.5)
    ax1.fill_between(timestamps, cpu_usage, alpha=0.3, color='blue')
    ax1.set_ylabel('CPU Usage (%)', fontsize=12)
    ax1.set_title('Resource Usage Over Time', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Memory usage
    ax2.plot(timestamps, memory_usage, color='green', linewidth=1.5)
    ax2.fill_between(timestamps, memory_usage, alpha=0.3, color='green')
    ax2.set_ylabel('Memory Usage (%)', fontsize=12)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor,
                         class_names: list, 
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix.
    
    Args:
        predictions: Prediction tensor [N, num_classes]
        labels: True labels [N]
        class_names: List of class names
        save_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    pred_labels = torch.argmax(predictions, dim=1).cpu().numpy()
    true_labels = labels.cpu().numpy()
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
