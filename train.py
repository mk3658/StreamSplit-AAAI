"""
Main training script for StreamSplit system.
Coordinates edge and server training.
"""

import torch
import torch.nn as nn
import torch    # Load config
    config = load_config(args.config)
    
    # Auto-detect and set best available device
    device = get_device(force_cpu=args.force_cpu)
    print_device_info(device)
    optimize_for_device(device)
    
    # Update config with detected device
    config['experiment']['device'] = str(device)
    
    # Create directories
    os.makedirs(config['experiment']['log_dir'], exist_ok=True)
    os.makedirs(config['experiment']['checkpoint_dir'], exist_ok=True) optim
from torch.utils.data import DataLoader
import yaml
import argparse
import os
from pathlib import Path

from models.mobilenet_v3 import MobileNetV3Small
from edge.audio_processing import AudioProcessor, AudioAugmentor, AdaptiveFeatureExtractor
from edge.contrastive_learning import StreamingContrastiveLearning
from edge.memory_bank import MemoryBank
from edge.resource_monitor import ResourceMonitor
from server.hybrid_loss import HybridLoss
from server.aggregation import ServerAggregator, UncertaintyEstimator
from utils.logger import Logger
from utils.metrics import MetricsTracker
from utils.device import get_device, print_device_info, optimize_for_device
from datasets import create_audioset_loaders


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_edge(config):
    """Setup edge device components."""
    # Resource monitor
    resource_monitor = ResourceMonitor(config)
    resource_monitor.start()
    
    # Audio processor
    audio_processor = AudioProcessor(config)
    audio_augmentor = AudioAugmentor(config)
    
    # Memory bank
    memory_bank = MemoryBank(
        min_size=config['edge']['memory_bank']['min_size'],
        max_size=config['edge']['memory_bank']['max_size'],
        embedding_dim=config['edge']['memory_bank']['embedding_dim'],
        num_components=config['edge']['das']['num_components'],
        alpha=config['edge']['das']['alpha']
    )
    
    # Encoder
    encoder = MobileNetV3Small(
        width_mult=config['server']['encoder']['width_mult'],
        embedding_dim=config['server']['encoder']['embedding_dim'],
        use_early_exit=config['server']['encoder']['use_early_exit']
    )
    
    # Contrastive learning module
    contrastive_module = StreamingContrastiveLearning(
        config, encoder, memory_bank
    )
    
    return {
        'resource_monitor': resource_monitor,
        'audio_processor': audio_processor,
        'audio_augmentor': audio_augmentor,
        'memory_bank': memory_bank,
        'encoder': encoder,
        'contrastive_module': contrastive_module
    }


def setup_server(config):
    """Setup server components."""
    # Hybrid loss
    hybrid_loss = HybridLoss(config)
    
    # Uncertainty estimator
    uncertainty_estimator = UncertaintyEstimator(
        config,
        num_prototypes=config['server']['prototypes']['num_centers']
    )
    
    # Aggregator
    aggregator = ServerAggregator(config)
    
    return {
        'hybrid_loss': hybrid_loss,
        'uncertainty_estimator': uncertainty_estimator,
        'aggregator': aggregator
    }


def train_epoch(edge_components, server_components, dataloader, 
                optimizer, device, config, epoch):
    """Train for one epoch."""
    edge_components['contrastive_module'].train()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (mel_specs, labels) in enumerate(dataloader):
        # Dataset now returns mel-spectrograms [batch, 1, 128, 1001]
        mel_specs = mel_specs.to(device)
        
        # Create augmented version for contrastive learning
        # Apply time/frequency masking as augmentation
        B, C, H, W = mel_specs.shape
        mel_specs_aug = mel_specs.clone()
        
        # Simple augmentation: add small noise
        noise = torch.randn_like(mel_specs_aug) * 0.1
        mel_specs_aug = mel_specs_aug + noise
        
        # Forward pass
        loss, info = edge_components['contrastive_module'](
            mel_specs, mel_specs_aug
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)
    
    # Auto-detect and set best available device
    device = get_device(force_cpu=args.force_cpu)
    print_device_info(device)
    optimize_for_device(device)
    
    # Update config with detected device
    config['experiment']['device'] = str(device)
    
    # Create directories
    os.makedirs(config['experiment']['log_dir'], exist_ok=True)
    os.makedirs(config['experiment']['checkpoint_dir'], exist_ok=True)
    
    # Setup components
    print("Setting up edge components...")
    edge_components = setup_edge(config)
    
    print("Setting up server components...")
    server_components = setup_server(config)
    
    # Move to device
    edge_components['contrastive_module'].to(device)
    server_components['hybrid_loss'].to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(
        edge_components['contrastive_module'].parameters(),
        lr=config['server']['training']['learning_rate'],
        weight_decay=config['server']['training']['weight_decay']
    )
    
    # Create datasets
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_audioset_loaders(config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Training loop
    num_epochs = config['server']['training']['num_epochs']
    
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train epoch with real data
        avg_loss = train_epoch(edge_components, server_components,
                              train_loader, optimizer, device, config, epoch)
        
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['logging']['save_frequency'] == 0:
            checkpoint_path = os.path.join(
                config['experiment']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch + 1}.pth'
            )
            # Create directory if needed
            os.makedirs(config['experiment']['checkpoint_dir'], exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': edge_components['encoder'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train StreamSplit model with GPU support'
    )
    parser.add_argument('--config', type=str, 
                       default='configs/streamsplit.yaml',
                       help='Path to configuration file')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    main(args)
