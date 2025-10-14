"""
Comprehensive Training Diagnostics for StreamSplit
Identifies issues with high loss and unstable training
"""

import torch
import torch.nn.functional as F
import yaml
import numpy as np
from pathlib import Path

from models import MobileNetV3Small
from datasets import AudioSetDataset, EdgeAudioDataset
from edge import (AudioProcessor, AudioAugmentor, MemoryBank, 
                 StreamingContrastiveLearning)
from server import HybridLoss


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_config():
    """Check configuration parameters."""
    print_section("1. Configuration Check")
    
    with open('configs/streamsplit.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n🔍 Critical Hyperparameters:")
    temp = config['edge']['contrastive']['temperature']
    lr = config['server']['training']['learning_rate']
    batch_size = config['server']['training']['batch_size']
    embedding_dim = config['edge']['memory_bank']['embedding_dim']
    
    print(f"  Temperature:           {temp}")
    print(f"  Learning Rate:         {lr}")
    print(f"  Batch Size:            {batch_size}")
    print(f"  Embedding Dim:         {embedding_dim}")
    print(f"  Memory Bank Size:      {config['edge']['memory_bank']['min_size']}-{config['edge']['memory_bank']['max_size']}")
    
    # Check for issues
    issues = []
    if temp < 0.05 or temp > 0.5:
        issues.append(f"⚠️  Temperature {temp} is unusual (0.07-0.2)")
    
    if batch_size < 256:
        issues.append(f"⚠️  Batch size {batch_size} is small (256+ recommended)")
    
    if issues:
        print("\n⚠️  Potential Issues Found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ Configuration looks reasonable")
    
    return config


def check_model_architecture(config):
    """Check if model is properly initialized."""
    print_section("2. Model Architecture Check")
    
    try:
        model = MobileNetV3Small(
            width_mult=config['server']['encoder']['width_mult'],
            embedding_dim=config['server']['encoder']['embedding_dim'],
            use_early_exit=config['server']['encoder']['use_early_exit']
        )
        
        print(f"\n✓ Model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Test forward pass
        dummy_input = torch.randn(4, 1, 128, 1000)  # Mel-spectrogram shape
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"\n✓ Forward pass successful")
        print(f"  Input shape:  {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"  Output mean:  {output.mean():.4f}")
        print(f"  Output std:   {output.std():.4f}")
        
        # Check for issues
        if output.mean().abs() > 10:
            print("  ⚠️  Output mean is very large - might need normalization")
        if output.std() < 0.1:
            print("  ⚠️  Output std is very small - model might not be learning")
        if torch.isnan(output).any():
            print("  ❌ NaN detected in output!")
        
        return model
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return None


def check_dataset(config):
    """Check dataset and data loading."""
    print_section("3. Dataset Check")
    
    try:
        # Try to load dataset
        dataset = AudioSetDataset(
            data_dir=config['data']['data_dir'],
            split='train',
            sample_rate=config['data']['sample_rate'],
            duration=config['data']['audio_duration'],
            num_classes=config['data']['num_classes']
        )
        
        print(f"\n✓ Dataset loaded successfully")
        print(f"  Total samples: {len(dataset)}")
        
        if len(dataset) < 100:
            print(f"  ⚠️  Very small dataset ({len(dataset)} samples)")
            print(f"      Contrastive learning needs 1000+ samples ideally")
        
        # Check a sample
        print("\n🔍 Checking sample data...")
        sample = dataset[0]
        
        if isinstance(sample, tuple):
            mel_spec = sample[0]
            print(f"  Sample type: tuple with {len(sample)} elements")
        else:
            mel_spec = sample
            print(f"  Sample type: {type(sample)}")
        
        print(f"  Mel-spec shape: {mel_spec.shape}")
        print(f"  Mel-spec dtype: {mel_spec.dtype}")
        print(f"  Mel-spec range: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
        print(f"  Mel-spec mean:  {mel_spec.mean():.4f}")
        print(f"  Mel-spec std:   {mel_spec.std():.4f}")
        
        # Check for data issues
        issues = []
        if torch.isnan(mel_spec).any():
            issues.append("❌ NaN values in data!")
        if torch.isinf(mel_spec).any():
            issues.append("❌ Inf values in data!")
        if mel_spec.std() < 0.01:
            issues.append("⚠️  Very low variance - data might not be normalized")
        if mel_spec.mean().abs() > 10:
            issues.append("⚠️  Large mean - data might not be centered")
        
        if issues:
            print("\n⚠️  Data Issues Found:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("\n✓ Data looks reasonable")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_contrastive_loss(config, model):
    """Test contrastive loss computation."""
    print_section("4. Contrastive Loss Check")
    
    try:
        # Create dummy embeddings
        batch_size = 32
        embedding_dim = config['edge']['memory_bank']['embedding_dim']
        temp = config['edge']['contrastive']['temperature']
        
        print(f"\n🔍 Testing with:")
        print(f"  Batch size: {batch_size}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Temperature: {temp}")
        
        # Case 1: Random embeddings (should give high loss)
        z1 = torch.randn(batch_size, embedding_dim)
        z2 = torch.randn(batch_size, embedding_dim)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        sim = torch.mm(z1, z2.T) / temp
        labels = torch.arange(batch_size)
        loss_random = F.cross_entropy(sim, labels)
        
        print(f"\n📊 Random embeddings:")
        print(f"  Similarity matrix range: [{sim.min():.4f}, {sim.max():.4f}]")
        print(f"  Loss: {loss_random.item():.4f}")
        print(f"  Expected: ~{np.log(batch_size):.2f} (log of batch size)")
        
        # Case 2: Perfect positive pairs (should give low loss)
        z1 = torch.randn(batch_size, embedding_dim)
        z2 = z1.clone()  # Perfect matches
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        sim = torch.mm(z1, z2.T) / temp
        loss_perfect = F.cross_entropy(sim, labels)
        
        print(f"\n📊 Perfect positive pairs:")
        print(f"  Diagonal similarity: {sim.diag().mean():.4f}")
        print(f"  Loss: {loss_perfect.item():.4f}")
        print(f"  Expected: ~0.0 (should be very low)")
        
        # Case 3: Realistic model embeddings
        if model is not None:
            dummy_input = torch.randn(batch_size, 1, 128, 1000)
            with torch.no_grad():
                z1 = model(dummy_input)
                z2 = model(dummy_input + 0.1 * torch.randn_like(dummy_input))
            
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            
            sim = torch.mm(z1, z2.T) / temp
            loss_model = F.cross_entropy(sim, labels)
            
            print(f"\n📊 Model embeddings (with slight augmentation):")
            print(f"  Positive pair similarity: {(z1 * z2).sum(1).mean():.4f}")
            print(f"  Loss: {loss_model.item():.4f}")
        
        # Analysis
        print("\n🔍 Analysis:")
        if loss_random.item() < np.log(batch_size) - 1:
            print("  ⚠️  Random loss is too low - temperature might be wrong")
        elif loss_random.item() > np.log(batch_size) + 2:
            print("  ⚠️  Random loss is too high - temperature might be wrong")
        else:
            print("  ✓ Random loss is in expected range")
        
        if loss_perfect.item() > 1.0:
            print("  ⚠️  Perfect pair loss is too high - something's wrong!")
        else:
            print("  ✓ Perfect pair loss is low (good)")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_gradient_flow(config, model, dataset):
    """Check if gradients are flowing properly."""
    print_section("5. Gradient Flow Check")
    
    if model is None or dataset is None:
        print("⚠️  Skipping - model or dataset not available")
        return
    
    try:
        # Create data loader
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=True, num_workers=0
        )
        
        # Get one batch
        batch = next(iter(loader))
        if isinstance(batch, tuple):
            mel_spec = batch[0]
            if len(batch) > 2:
                mel_spec_aug = batch[1]
            else:
                mel_spec_aug = mel_spec
        else:
            mel_spec = batch
            mel_spec_aug = mel_spec
        
        print(f"\n✓ Got batch: {mel_spec.shape}")
        
        # Forward pass
        model.train()
        z1 = model(mel_spec)
        z2 = model(mel_spec_aug)
        
        # Compute loss
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        temp = config['edge']['contrastive']['temperature']
        sim = torch.mm(z1, z2.T) / temp
        labels = torch.arange(z1.size(0))
        loss = F.cross_entropy(sim, labels)
        
        print(f"\n📊 Forward pass:")
        print(f"  Embedding shape: {z1.shape}")
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_norms = []
        zero_grads = 0
        nan_grads = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if grad_norm == 0:
                    zero_grads += 1
                if torch.isnan(param.grad).any():
                    nan_grads += 1
        
        print(f"\n🔍 Gradient analysis:")
        print(f"  Parameters with gradients: {len(grad_norms)}")
        print(f"  Zero gradients: {zero_grads}")
        print(f"  NaN gradients: {nan_grads}")
        if grad_norms:
            print(f"  Mean gradient norm: {np.mean(grad_norms):.6f}")
            print(f"  Max gradient norm: {np.max(grad_norms):.6f}")
            print(f"  Min gradient norm: {np.min(grad_norms):.6f}")
        
        # Check for issues
        if nan_grads > 0:
            print("  ❌ NaN gradients detected - training will fail!")
        elif zero_grads > len(grad_norms) * 0.5:
            print("  ⚠️  Many zero gradients - model might not be learning")
        elif np.mean(grad_norms) < 1e-7:
            print("  ⚠️  Very small gradients - learning rate might be too low")
        elif np.max(grad_norms) > 100:
            print("  ⚠️  Very large gradients - might need gradient clipping")
        else:
            print("  ✓ Gradients look reasonable")
        
    except Exception as e:
        print(f"❌ Gradient check failed: {e}")
        import traceback
        traceback.print_exc()


def check_checkpoint_loss_pattern():
    """Analyze loss pattern from checkpoints."""
    print_section("6. Checkpoint Loss Pattern Analysis")
    
    try:
        import json
        with open('evaluation_results.json', 'r') as f:
            results = json.load(f)
        
        checkpoints = sorted(results['checkpoints'], key=lambda x: x['epoch'])
        
        print("\n📈 Loss progression:")
        losses = []
        for cp in checkpoints:
            epoch = cp['epoch'] + 1
            loss = cp['train_loss']
            losses.append(loss)
            print(f"  Epoch {epoch:3d}: {loss:.4f}")
        
        # Calculate statistics
        print(f"\n📊 Statistics:")
        print(f"  Mean loss: {np.mean(losses):.4f}")
        print(f"  Std dev: {np.std(losses):.4f}")
        print(f"  Min loss: {np.min(losses):.4f}")
        print(f"  Max loss: {np.max(losses):.4f}")
        
        # Check for issues
        print(f"\n🔍 Pattern analysis:")
        
        # Check variance in last 30%
        last_30_pct = int(len(losses) * 0.7)
        late_losses = losses[last_30_pct:]
        late_std = np.std(late_losses)
        
        if late_std > 0.3:
            print(f"  ⚠️  High variance in late training ({late_std:.4f})")
            print(f"      Model hasn't converged properly")
        
        # Check for improvement
        if losses[-1] > losses[0]:
            print(f"  ⚠️  Loss increased from start to end")
        elif losses[-1] > np.min(losses) * 1.1:
            print(f"  ⚠️  Final loss not close to best loss")
        
        # Check for oscillations
        changes = [abs(losses[i] - losses[i-1]) for i in range(1, len(losses))]
        large_changes = sum(1 for c in changes if c > 0.3)
        
        if large_changes > len(changes) * 0.3:
            print(f"  ⚠️  Many large loss jumps ({large_changes}/{len(changes)})")
            print(f"      Learning rate might be too high")
        
        # Overall assessment
        if np.min(losses) > 2.0:
            print(f"\n❌ CRITICAL: Minimum loss ({np.min(losses):.4f}) is very high!")
            print(f"   Expected for contrastive learning: < 1.5")
            print(f"   This suggests fundamental training issues")
        
    except Exception as e:
        print(f"⚠️  Could not analyze checkpoints: {e}")


def run_mini_training_test(config, model, dataset):
    """Run a few training steps to see if loss decreases."""
    print_section("7. Mini Training Test (10 steps)")
    
    if model is None or dataset is None:
        print("⚠️  Skipping - model or dataset not available")
        return
    
    try:
        # Setup
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True, num_workers=0
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        temp = config['edge']['contrastive']['temperature']
        
        print("\n🏃 Running 10 training steps...")
        model.train()
        losses = []
        
        for step, batch in enumerate(loader):
            if step >= 10:
                break
            
            if isinstance(batch, tuple):
                mel_spec = batch[0]
                mel_spec_aug = batch[1] if len(batch) > 2 else mel_spec
            else:
                mel_spec = batch
                mel_spec_aug = mel_spec
            
            optimizer.zero_grad()
            
            # Forward
            z1 = model(mel_spec)
            z2 = model(mel_spec_aug)
            
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            
            sim = torch.mm(z1, z2.T) / temp
            labels = torch.arange(z1.size(0))
            loss = F.cross_entropy(sim, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            print(f"  Step {step+1}: loss = {loss.item():.4f}")
        
        # Analysis
        print(f"\n📊 Results:")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Change: {losses[-1] - losses[0]:.4f}")
        
        if losses[-1] < losses[0]:
            improvement = (losses[0] - losses[-1]) / losses[0] * 100
            print(f"  ✓ Loss decreased by {improvement:.2f}% (good!)")
        else:
            print(f"  ❌ Loss increased - model not learning!")
        
        if losses[0] > 5.0:
            print(f"  ⚠️  Initial loss very high ({losses[0]:.4f})")
        
    except Exception as e:
        print(f"❌ Mini training test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all diagnostic checks."""
    print("\n" + "=" * 80)
    print("  STREAMSPLIT TRAINING DIAGNOSTICS")
    print("=" * 80)
    print("\nThis will identify why training loss is high and unstable.\n")
    
    # Run checks
    config = check_config()
    model = check_model_architecture(config)
    dataset = check_dataset(config)
    check_contrastive_loss(config, model)
    check_gradient_flow(config, model, dataset)
    check_checkpoint_loss_pattern()
    run_mini_training_test(config, model, dataset)
    
    # Final summary
    print_section("SUMMARY & RECOMMENDATIONS")
    
    print("""
Based on the diagnostics above, common issues and fixes:

1. ❌ If loss > 3.0 throughout training:
   → Temperature too high or batch size too small
   → Try: temperature=0.07, batch_size=256

2. ❌ If loss oscillates wildly (>30% jumps):
   → Learning rate too high
   → Try: reduce lr by 10x or add lr scheduler

3. ❌ If gradients are NaN or zero:
   → Data normalization issue or model architecture problem
   → Check data preprocessing and model initialization

4. ❌ If mini-training doesn't decrease loss:
   → Fundamental training setup issue
   → Review loss computation and data augmentation

5. ✓ If loss ~0.5-1.5 and decreasing:
   → Training is working correctly!

Next steps:
1. Review the diagnostics above
2. Fix any identified issues
3. Retrain with corrected parameters
4. Expected result: loss should reach < 1.5
    """)
    
    print("=" * 80)
    print("\n✅ Diagnostics complete!\n")


if __name__ == '__main__':
    main()
