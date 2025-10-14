"""
Device utilities for GPU/CPU selection.

Provides automatic device detection and selection for training.
"""

import torch
from typing import Optional


def get_device(force_cpu: bool = False, verbose: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Priority: CUDA > MPS > CPU
    
    Args:
        force_cpu: Force CPU usage even if GPU is available
        verbose: Print device information
        
    Returns:
        torch.device: Selected device
    """
    if force_cpu:
        device = torch.device('cpu')
        if verbose:
            print("⚠️  Forcing CPU usage (GPU disabled by user)")
        return device
    
    # Check CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            print(f"✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Available GPUs: {torch.cuda.device_count()}")
        return device
    
    # Check MPS (Apple Silicon GPU)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        if verbose:
            print("✓ Using Apple MPS (Metal Performance Shaders)")
        return device
    
    # Fallback to CPU
    device = torch.device('cpu')
    if verbose:
        print("⚠️  No GPU detected, using CPU")
        print("  For faster training:")
        print("  - Use Google Colab with GPU runtime")
        print("  - Or AWS/GCP instances with NVIDIA GPUs")
    return device


def print_device_info(device: torch.device):
    """
    Print detailed device information.
    
    Args:
        device: torch.device to print info about
    """
    print("\n" + "=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Device Type: {device.type}")
    
    if device.type == 'cuda':
        print(f"\nCUDA Details:")
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        
        # Memory info
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\nMemory:")
            print(f"  Total: {total_memory:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Free: {total_memory - reserved:.2f} GB")
    
    elif device.type == 'mps':
        print("\nApple Silicon GPU (Metal Performance Shaders)")
        print("  Note: Some operations may fall back to CPU")
    
    else:
        print("\nCPU Mode")
        print(f"  PyTorch built with CUDA: {torch.version.cuda is not None}")
        import multiprocessing
        print(f"  Available CPU cores: {multiprocessing.cpu_count()}")
    
    print("=" * 60 + "\n")


def optimize_for_device(device: torch.device):
    """
    Apply device-specific optimizations.
    
    Args:
        device: torch.device to optimize for
    """
    if device.type == 'cuda':
        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        print("✓ Enabled cuDNN benchmark mode for faster training")
        
        # Enable TF32 for A100 GPUs (faster without precision loss)
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✓ Enabled TF32 for faster computation on Ampere GPU")
    
    elif device.type == 'mps':
        # MPS-specific optimizations
        print("✓ Using MPS optimizations for Apple Silicon")
    
    else:
        # CPU optimizations
        print("✓ Using CPU optimizations")


def get_device_memory_info(device: torch.device) -> dict:
    """
    Get memory information for the device.
    
    Args:
        device: torch.device
        
    Returns:
        Dictionary with memory information
    """
    if device.type == 'cuda':
        return {
            'total': torch.cuda.get_device_properties(0).total_memory,
            'allocated': torch.cuda.memory_allocated(0),
            'reserved': torch.cuda.memory_reserved(0),
            'free': torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
        }
    else:
        return {
            'total': 0,
            'allocated': 0,
            'reserved': 0,
            'free': 0
        }


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")


def check_gpu_availability() -> dict:
    """
    Check GPU availability and return status.
    
    Returns:
        Dictionary with GPU availability information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if info['cuda_available']:
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(info['num_gpus'])]
    
    return info
