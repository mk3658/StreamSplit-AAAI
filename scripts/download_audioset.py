"""
Download and prepare AudioSet dataset.
This script creates a synthetic AudioSet subset for demonstration.
For production, integrate with YouTube download using yt-dlp.
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import create_audioset_loaders


def main():
    parser = argparse.ArgumentParser(
        description='Download and prepare AudioSet dataset'
    )
    parser.add_argument('--config', type=str,
                       default='configs/streamsplit.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes to use')
    parser.add_argument('--subset', type=str, default='balanced',
                       choices=['balanced', 'full'],
                       help='AudioSet subset to download')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config
    config['data']['num_classes'] = args.num_classes
    
    print(f"\nDownloading AudioSet ({args.subset} subset)...")
    print(f"Number of classes: {args.num_classes}")
    print(f"Data directory: {config['data']['data_dir']}")
    
    # Create dataloaders (this will trigger download/creation)
    print("\nCreating datasets...")
    train_loader, val_loader, test_loader = create_audioset_loaders(config)
    
    print(f"\n{'='*60}")
    print("Dataset Statistics:")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Total samples: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)}")
    
    # Show class mapping
    class_mapping = train_loader.dataset.get_class_mapping()
    print(f"\nClass Mapping:")
    for class_id, class_name in sorted(class_mapping.items(), key=lambda x: int(x[0])):
        print(f"  {class_id}: {class_name}")
    
    # Test loading a batch
    print(f"\nTesting data loading...")
    waveforms, labels = next(iter(train_loader))
    print(f"  Batch shape: {waveforms.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Waveform range: [{waveforms.min():.3f}, {waveforms.max():.3f}]")
    
    print(f"\n{'='*60}")
    print("✓ AudioSet preparation completed successfully!")
    print(f"{'='*60}")
    
    print(f"\nNote: This is synthetic data for demonstration.")
    print(f"For production use, implement YouTube download with yt-dlp:")
    print(f"  pip install yt-dlp")
    print(f"  # Modify datasets/audioset.py _download() method")


if __name__ == '__main__':
    main()
