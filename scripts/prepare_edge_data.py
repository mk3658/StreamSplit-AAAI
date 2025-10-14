"""
Prepare edge audio dataset from recorded audio streams.
Creates synthetic edge audio data for demonstration.
"""

import argparse
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets import create_edge_loaders


def main():
    parser = argparse.ArgumentParser(
        description='Prepare edge audio dataset'
    )
    parser.add_argument('--config', type=str,
                       default='configs/streamsplit.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing audio files')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update data directory if provided
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    
    print("\nPreparing edge audio dataset...")
    print(f"Data directory: {config['data']['data_dir']}")
    
    # Create dataloaders (this will trigger dataset creation)
    print("\nCreating datasets...")
    train_loader, val_loader, test_loader = create_edge_loaders(config)
    
    print(f"\n{'='*60}")
    print("Dataset Statistics:")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    total_samples = (len(train_loader.dataset) + 
                    len(val_loader.dataset) + 
                    len(test_loader.dataset))
    print(f"Total samples: {total_samples}")
    
    # Show class mapping
    class_mapping = train_loader.dataset.get_class_mapping()
    print("\nClass Mapping (Smart Home/Urban Sounds):")
    for class_id, class_name in sorted(class_mapping.items(), 
                                       key=lambda x: int(x[0])):
        print(f"  {class_id}: {class_name}")
    
    # Test loading a batch
    print("\nTesting data loading...")
    batch = next(iter(train_loader))
    waveforms, labels, metadata = batch
    print(f"  Batch shape: {waveforms.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Waveform range: [{waveforms.min():.3f}, {waveforms.max():.3f}]")
    # metadata is a list of dicts, one per sample in batch
    if isinstance(metadata, list) and len(metadata) > 0:
        print(f"  Metadata fields: {list(metadata[0].keys())}")
        device_ids = [m['device_id'] for m in metadata[:3]]
        print(f"  Sample device IDs: {device_ids}")
    
    print(f"\n{'='*60}")
    print("✓ Edge audio dataset preparation completed successfully!")
    print(f"{'='*60}")
    
    print("\nNote: This is synthetic data for demonstration.")
    print("For production use with real edge devices:")
    print("  1. Record audio streams from Raspberry Pi")
    print("  2. Annotate with class labels")
    print("  3. Place in data_dir/edge_audio/audio/")
    print("  4. Update metadata JSON files")


if __name__ == '__main__':
    main()
