"""
Quick demo/example of StreamSplit components.
Run this to verify installation and basic functionality.
"""

import torch
import yaml
import numpy as np

from models import MobileNetV3Small
from edge import (AudioProcessor, AudioAugmentor, ResourceMonitor, 
                 MemoryBank, StreamingContrastiveLearning)
from server import HybridLoss, UncertaintyEstimator
from utils import Logger, MetricsTracker


def demo_audio_processing():
    """Demo audio processing pipeline."""
    print("\n=== Audio Processing Demo ===")
    
    # Load config
    with open('configs/streamsplit.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create processor
    processor = AudioProcessor(config)
    augmentor = AudioAugmentor(config)
    
    # Generate dummy waveform (10 seconds at 16kHz)
    sample_rate = config['data']['sample_rate']
    duration = config['data']['audio_duration']
    waveform = torch.randn(1, sample_rate * duration)
    
    # Process
    mel_spec = processor.process_audio(waveform)
    print(f"Input waveform shape: {waveform.shape}")
    print(f"Mel-spectrogram shape: {mel_spec.shape}")
    
    # Augment
    waveform_aug = augmentor.augment(waveform)
    print(f"Augmented waveform shape: {waveform_aug.shape}")
    
    print("✓ Audio processing works!")


def demo_model():
    """Demo model architecture."""
    print("\n=== Model Demo ===")
    
    # Create model
    model = MobileNetV3Small(
        width_mult=0.75,
        embedding_dim=128,
        use_early_exit=True
    )
    
    # Dummy input (mel-spectrogram)
    batch_size = 4
    n_mels = 128
    time_frames = 100
    x = torch.randn(batch_size, 1, n_mels, time_frames)
    
    # Forward pass
    embeddings = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    
    # Try early exit
    embeddings_early = model(x, use_early_exit=True)
    print(f"Early exit embeddings shape: {embeddings_early.shape}")
    
    print("✓ Model works!")


def demo_memory_bank():
    """Demo memory bank with DAS."""
    print("\n=== Memory Bank Demo ===")
    
    # Create memory bank
    bank = MemoryBank(
        min_size=64,
        max_size=512,
        embedding_dim=128,
        num_components=5
    )
    
    # Add embeddings
    for i in range(100):
        emb = torch.randn(128)
        bank.push(emb, timestamp=float(i))
    
    print(f"Memory bank size: {len(bank)}")
    
    # Sample negatives
    neg_emb, neg_weights = bank.sample(n_samples=32, use_das=True)
    print(f"Sampled negatives shape: {neg_emb.shape}")
    print(f"Negative weights shape: {neg_weights.shape}")
    
    print("✓ Memory bank works!")


def demo_hybrid_loss():
    """Demo hybrid loss function."""
    print("\n=== Hybrid Loss Demo ===")
    
    with open('configs/streamsplit.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create loss
    loss_fn = HybridLoss(config)
    
    # Dummy embeddings
    edge_emb = torch.randn(32, 128)
    server_emb = torch.randn(32, 128)
    
    # Compute loss
    loss = loss_fn(edge_emb, server_emb, return_components=True)
    
    print(f"Total loss: {loss['loss_total']:.4f}")
    print(f"SW loss: {loss['loss_sw']:.4f}")
    print(f"Laplacian loss: {loss['loss_lap']:.4f}")
    
    print("✓ Hybrid loss works!")


def demo_resource_monitor():
    """Demo resource monitoring."""
    print("\n=== Resource Monitor Demo ===")
    
    with open('configs/streamsplit.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create monitor
    monitor = ResourceMonitor(config)
    monitor.start()
    
    # Wait a bit and get state
    import time
    time.sleep(0.5)
    
    state = monitor.get_state()
    print(f"CPU utilization: {state['cpu_util']*100:.1f}%")
    print(f"Memory usage: {state['mem_usage']*100:.1f}%")
    print(f"Battery level: {state['battery_level']*100:.1f}%")
    
    monitor.stop()
    print("✓ Resource monitor works!")


def main():
    """Run all demos."""
    print("=" * 60)
    print("StreamSplit Demo - Verifying Installation")
    print("=" * 60)
    
    try:
        demo_audio_processing()
        demo_model()
        demo_memory_bank()
        demo_hybrid_loss()
        demo_resource_monitor()
        
        print("\n" + "=" * 60)
        print("✓ All demos passed! Installation is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
