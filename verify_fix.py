"""
Verify that dataset now returns correct mel-spectrogram format.
Run this before retraining to confirm the fix works.
"""

import torch
import yaml
from datasets import AudioSetDataset, EdgeAudioDataset

print("=" * 80)
print("DATASET FIX VERIFICATION")
print("=" * 80)

# Load config
with open('configs/streamsplit.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("\n1. Testing AudioSet Dataset")
print("-" * 80)

try:
    dataset = AudioSetDataset(
        data_dir=config['data']['data_dir'],
        split='train',
        sample_rate=config['data']['sample_rate'],
        duration=config['data']['audio_duration'],
        num_classes=config['data']['num_classes'],
        n_fft=config['edge']['fft']['n_fft'],
        hop_length=int(config['edge']['fft']['hop_size_ms'] * 
                       config['data']['sample_rate'] / 1000),
        n_mels=config['edge']['fft']['n_mels']
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Test a sample
    sample = dataset[0]
    mel_spec, label = sample
    
    print(f"\n✓ Sample loaded successfully")
    print(f"  Mel-spec shape: {mel_spec.shape}")
    print(f"  Expected shape: [1, 128, ~1000] (channels, mels, time)")
    print(f"  Mel-spec dtype: {mel_spec.dtype}")
    print(f"  Mel-spec range: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
    print(f"  Mel-spec mean:  {mel_spec.mean():.4f}")
    print(f"  Mel-spec std:   {mel_spec.std():.4f}")
    print(f"  Label: {label}")
    
    # Verify shape
    if len(mel_spec.shape) == 3:
        channels, n_mels, n_frames = mel_spec.shape
        if channels == 1 and n_mels == 128:
            print(f"\n✅ CORRECT FORMAT!")
            print(f"   Shape {mel_spec.shape} matches expected [1, 128, ~1000]")
        else:
            print(f"\n⚠️  Shape issue: got {mel_spec.shape}")
    else:
        print(f"\n❌ WRONG FORMAT!")
        print(f"   Expected 3D tensor [channels, mels, time]")
        print(f"   Got {len(mel_spec.shape)}D tensor: {mel_spec.shape}")
    
    # Check normalization
    if abs(mel_spec.mean()) < 0.5 and 0.5 < mel_spec.std() < 2.0:
        print(f"✅ Data is well-normalized (mean≈0, std≈1)")
    else:
        print(f"⚠️  Data normalization may need adjustment")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n2. Testing EdgeAudio Dataset")
print("-" * 80)

try:
    dataset = EdgeAudioDataset(
        data_dir=config['data']['data_dir'],
        split='train',
        sample_rate=config['data']['sample_rate'],
        duration=config['data']['audio_duration'],
        num_classes=7,
        n_fft=config['edge']['fft']['n_fft'],
        hop_length=int(config['edge']['fft']['hop_size_ms'] * 
                       config['data']['sample_rate'] / 1000),
        n_mels=config['edge']['fft']['n_mels']
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Test a sample
    sample = dataset[0]
    mel_spec, label, metadata = sample
    
    print(f"\n✓ Sample loaded successfully")
    print(f"  Mel-spec shape: {mel_spec.shape}")
    print(f"  Expected shape: [1, 128, ~1000]")
    print(f"  Mel-spec range: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
    print(f"  Label: {label}")
    print(f"  Metadata: {metadata}")
    
    # Verify shape
    if len(mel_spec.shape) == 3 and mel_spec.shape[0] == 1 and mel_spec.shape[1] == 128:
        print(f"\n✅ CORRECT FORMAT!")
    else:
        print(f"\n❌ WRONG FORMAT: {mel_spec.shape}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing Model Forward Pass")
print("-" * 80)

try:
    from models import MobileNetV3Small
    
    model = MobileNetV3Small(
        width_mult=config['server']['encoder']['width_mult'],
        embedding_dim=config['server']['encoder']['embedding_dim'],
        use_early_exit=config['server']['encoder']['use_early_exit']
    )
    
    print("✓ Model created")
    
    # Test with real data (EdgeAudio returns 3 values)
    sample = dataset[0]
    if len(sample) == 3:
        mel_spec, label, metadata = sample
    else:
        mel_spec, label = sample
    
    # Add batch dimension
    mel_spec_batch = mel_spec.unsqueeze(0)  # [1, 1, 128, time]
    
    print(f"  Input shape to model: {mel_spec_batch.shape}")
    
    with torch.no_grad():
        embeddings = model(mel_spec_batch)
    
    print(f"  Output embeddings shape: {embeddings.shape}")
    print(f"  Output embeddings range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    print(f"  Output embeddings std: {embeddings.std():.4f}")
    
    if embeddings.shape == (1, 128):
        print(f"\n✅ MODEL FORWARD PASS SUCCESSFUL!")
        if embeddings.std() > 0.01:
            print(f"✅ Embeddings have good variance (std={embeddings.std():.4f})")
        else:
            print(f"⚠️  Embeddings have low variance (std={embeddings.std():.4f})")
    else:
        print(f"\n❌ Unexpected output shape: {embeddings.shape}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

print("""
✅ If all checks passed:
   → Dataset is fixed and ready for training!
   → Delete old checkpoints: rm checkpoints/*.pth
   → Start fresh training: python train.py --config configs/streamsplit.yaml
   → Expected: Loss should reach < 1.0 by epoch 50

❌ If any checks failed:
   → Review error messages above
   → Check dataset implementation
   → Ensure mel-spectrogram transform is properly initialized
""")

print("=" * 80)
