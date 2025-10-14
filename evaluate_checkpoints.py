"""
Comprehensive evaluation script for all StreamSplit checkpoints.
Evaluates model performance across different training epochs.
"""

import torch
import yaml
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from models import MobileNetV3Small
from datasets import AudioSetDataset, EdgeAudioDataset
from edge import AudioProcessor, MemoryBank, StreamingContrastiveLearning
from server import HybridLoss, UncertaintyEstimator
from utils import MetricsTracker, Logger


def load_checkpoint(checkpoint_path, model, device):
    """Load model weights from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
    else:
        model.load_state_dict(checkpoint)
        epoch = 'unknown'
        loss = 'unknown'
    
    return model, epoch, loss


def evaluate_checkpoint(model, dataloader, config, device):
    """Evaluate a single checkpoint on the test set."""
    model.eval()
    
    # Initialize components
    contrastive_loss = StreamingContrastiveLearning(config).to(device)
    hybrid_loss = HybridLoss(config).to(device)
    uncertainty_estimator = UncertaintyEstimator(config).to(device)
    
    metrics = {
        'total_loss': 0.0,
        'contrastive_loss': 0.0,
        'hybrid_loss': 0.0,
        'num_batches': 0,
        'embeddings_norm': [],
        'feature_diversity': [],
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                # Unpack batch
                if len(batch) == 3:
                    mel_spec, mel_spec_aug, labels = batch
                else:
                    mel_spec, labels = batch[:2]
                    mel_spec_aug = mel_spec
                
                mel_spec = mel_spec.to(device)
                mel_spec_aug = mel_spec_aug.to(device)
                
                # Forward pass
                embeddings = model(mel_spec)
                embeddings_aug = model(mel_spec_aug)
                
                # Compute losses
                try:
                    loss_contrastive = contrastive_loss(embeddings, embeddings_aug)
                    loss_hybrid = hybrid_loss(embeddings, embeddings_aug)
                    total_loss = loss_contrastive + loss_hybrid
                    
                    metrics['contrastive_loss'] += loss_contrastive.item()
                    metrics['hybrid_loss'] += loss_hybrid.item()
                    metrics['total_loss'] += total_loss.item()
                except Exception as e:
                    print(f"Warning: Loss computation failed for batch {batch_idx}: {e}")
                    continue
                
                # Compute embedding statistics
                embedding_norm = torch.norm(embeddings, dim=1).mean().item()
                metrics['embeddings_norm'].append(embedding_norm)
                
                # Feature diversity (std of embeddings)
                feature_std = embeddings.std(dim=0).mean().item()
                metrics['feature_diversity'].append(feature_std)
                
                metrics['num_batches'] += 1
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Average metrics
    if metrics['num_batches'] > 0:
        metrics['avg_total_loss'] = metrics['total_loss'] / metrics['num_batches']
        metrics['avg_contrastive_loss'] = metrics['contrastive_loss'] / metrics['num_batches']
        metrics['avg_hybrid_loss'] = metrics['hybrid_loss'] / metrics['num_batches']
        metrics['avg_embedding_norm'] = np.mean(metrics['embeddings_norm'])
        metrics['avg_feature_diversity'] = np.mean(metrics['feature_diversity'])
    else:
        print("Warning: No batches were successfully processed!")
        metrics['avg_total_loss'] = float('inf')
        metrics['avg_contrastive_loss'] = float('inf')
        metrics['avg_hybrid_loss'] = float('inf')
        metrics['avg_embedding_norm'] = 0.0
        metrics['avg_feature_diversity'] = 0.0
    
    return metrics


def create_test_dataloader(config):
    """Create test dataloader."""
    dataset_name = config['data']['dataset']
    
    if dataset_name == 'audioset':
        test_dataset = AudioSetDataset(
            data_dir=config['data']['data_dir'],
            split='test',
            sample_rate=config['data']['sample_rate'],
            duration=config['data']['audio_duration'],
            num_classes=config['data']['num_classes']
        )
    else:
        test_dataset = EdgeAudioDataset(
            data_dir=config['data']['data_dir'],
            split='test',
            sample_rate=config['data']['sample_rate'],
            duration=config['data']['audio_duration'],
            num_classes=config['data']['num_classes']
        )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=False
    )
    
    return test_loader


def main():
    """Main evaluation pipeline."""
    print("=" * 70)
    print("StreamSplit Checkpoint Evaluation")
    print("=" * 70)
    
    # Load configuration
    config_path = 'configs/streamsplit.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['experiment']['device'])
    print(f"\nUsing device: {device}")
    
    # Find all checkpoints
    checkpoint_dir = Path(config['experiment']['checkpoint_dir'])
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoints to evaluate:")
    for cp in checkpoints:
        print(f"  - {cp.name}")
    
    # Create test dataloader
    print("\nCreating test dataloader...")
    try:
        test_loader = create_test_dataloader(config)
        print(f"Test set size: {len(test_loader.dataset)} samples")
        print(f"Number of batches: {len(test_loader)}")
    except Exception as e:
        print(f"Error creating test dataloader: {e}")
        print("Evaluation will be limited to checkpoint inspection only.")
        test_loader = None
    
    # Results storage
    results = {
        'config': config_path,
        'device': str(device),
        'evaluation_time': datetime.now().isoformat(),
        'checkpoints': []
    }
    
    # Evaluate each checkpoint
    for checkpoint_path in checkpoints:
        print("\n" + "=" * 70)
        print(f"Evaluating: {checkpoint_path.name}")
        print("=" * 70)
        
        try:
            # Create fresh model
            model = MobileNetV3Small(
                width_mult=config['server']['encoder']['width_mult'],
                embedding_dim=config['server']['encoder']['embedding_dim'],
                use_early_exit=config['server']['encoder']['use_early_exit']
            ).to(device)
            
            # Load checkpoint
            model, epoch, train_loss = load_checkpoint(str(checkpoint_path), model, device)
            
            checkpoint_result = {
                'checkpoint': checkpoint_path.name,
                'epoch': epoch,
                'train_loss': train_loss,
            }
            
            # Evaluate if test loader is available
            if test_loader is not None:
                metrics = evaluate_checkpoint(model, test_loader, config, device)
                checkpoint_result.update(metrics)
                
                print(f"\nResults for {checkpoint_path.name}:")
                print(f"  Epoch: {epoch}")
                print(f"  Train Loss: {train_loss}")
                print(f"  Test Loss: {metrics['avg_total_loss']:.4f}")
                print(f"  Contrastive Loss: {metrics['avg_contrastive_loss']:.4f}")
                print(f"  Hybrid Loss: {metrics['avg_hybrid_loss']:.4f}")
                print(f"  Avg Embedding Norm: {metrics['avg_embedding_norm']:.4f}")
                print(f"  Avg Feature Diversity: {metrics['avg_feature_diversity']:.4f}")
            else:
                print(f"  Epoch: {epoch}")
                print(f"  Train Loss: {train_loss}")
                print("  (Test evaluation skipped - no dataloader available)")
            
            results['checkpoints'].append(checkpoint_result)
            
        except Exception as e:
            print(f"Error evaluating {checkpoint_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    results_file = 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, indent=2, fp=f)
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    if test_loader is not None and results['checkpoints']:
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        
        # Find best checkpoint
        valid_checkpoints = [cp for cp in results['checkpoints'] 
                            if 'avg_total_loss' in cp and cp['avg_total_loss'] != float('inf')]
        
        if valid_checkpoints:
            best_checkpoint = min(valid_checkpoints, key=lambda x: x['avg_total_loss'])
            
            print(f"\nBest checkpoint: {best_checkpoint['checkpoint']}")
            print(f"  Test Loss: {best_checkpoint['avg_total_loss']:.4f}")
            print(f"  Epoch: {best_checkpoint['epoch']}")
            
            # Print all results in table format
            print("\n" + "-" * 70)
            print(f"{'Checkpoint':<25} {'Epoch':<8} {'Test Loss':<12} {'Contrastive':<12}")
            print("-" * 70)
            
            for cp in results['checkpoints']:
                if 'avg_total_loss' in cp:
                    print(f"{cp['checkpoint']:<25} {str(cp['epoch']):<8} "
                          f"{cp['avg_total_loss']:<12.4f} {cp['avg_contrastive_loss']:<12.4f}")
            
            print("-" * 70)


if __name__ == '__main__':
    main()
