"""
Audio processing module for edge devices.
Implements optimized FFT, adaptive feature extraction, and audio augmentations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from typing import Tuple, Optional, Dict
import time


class AudioProcessor:
    """Optimized audio processor for edge devices."""
    
    def __init__(self, config: Dict):
        """
        Initialize audio processor.
        
        Args:
            config: Configuration dictionary containing FFT and processing parameters
        """
        self.sample_rate = config['data']['sample_rate']
        self.fft_config = config['edge']['fft']
        self.reduced_config = config['edge']['reduced_resolution']
        self.adaptive_config = config['edge']['adaptive_policy']
        
        # FFT parameters
        self.window_size = int(self.fft_config['window_size_ms'] * self.sample_rate / 1000)
        self.hop_size = int(self.fft_config['hop_size_ms'] * self.sample_rate / 1000)
        self.n_fft = self.fft_config['n_fft']
        self.n_mels = self.fft_config['n_mels']
        
        # Reduced resolution parameters
        self.reduced_hop_size = int(self.reduced_config['hop_size_ms'] * self.sample_rate / 1000)
        self.reduced_n_mels = self.reduced_config['n_mels']
        
        # Create Hann window
        self.window = torch.hann_window(self.window_size, dtype=torch.float32)
        
        # Mel filterbanks
        self._create_mel_filterbank(self.n_mels, self.fft_config['fmin'], self.fft_config['fmax'])
        self._create_mel_filterbank(self.reduced_n_mels, self.reduced_config['fmin'], 
                                   self.reduced_config['fmax'], reduced=True)
        
        # Processing mode
        self.current_mode = 'full'  # 'full' or 'reduced'
        
    def _create_mel_filterbank(self, n_mels: int, fmin: float, fmax: float, reduced: bool = False):
        """Create mel-scale filterbank."""
        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        # Create mel scale
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to FFT bin numbers
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # Create filterbank
        filterbank = np.zeros((n_mels, self.n_fft // 2 + 1))
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            # Rising slope
            for j in range(left, center):
                filterbank[i, j] = (j - left) / (center - left)
            
            # Falling slope
            for j in range(center, right):
                filterbank[i, j] = (right - j) / (right - center)
        
        # Store as tensor
        if reduced:
            self.mel_filterbank_reduced = torch.from_numpy(filterbank).float()
        else:
            self.mel_filterbank = torch.from_numpy(filterbank).float()
    
    def process_audio(self, waveform: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        """
        Process raw audio waveform into mel-spectrogram.
        
        Args:
            waveform: Input audio tensor [batch_size, num_samples] or [num_samples]
            mode: Processing mode ('full' or 'reduced'). If None, uses current mode.
            
        Returns:
            Mel-spectrogram tensor [batch_size, n_mels, time_frames]
        """
        if mode is None:
            mode = self.current_mode
            
        # Ensure batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Select parameters based on mode
        if mode == 'full':
            hop_size = self.hop_size
            n_mels = self.n_mels
            mel_fb = self.mel_filterbank
        else:
            hop_size = self.reduced_hop_size
            n_mels = self.reduced_n_mels
            mel_fb = self.mel_filterbank_reduced
        
        # Compute STFT
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=hop_size,
            win_length=self.window_size,
            window=self.window,
            center=True,
            normalized=False,
            return_complex=True
        )
        
        # Magnitude spectrogram
        mag_spec = torch.abs(spec)  # [batch, freq_bins, time]
        
        # Apply mel filterbank
        mel_spec = torch.matmul(mel_fb, mag_spec)  # [batch, n_mels, time]
        
        # Log compression
        mel_spec = torch.log(mel_spec + 1e-9)
        
        return mel_spec
    
    def set_mode(self, mode: str):
        """Set processing mode."""
        assert mode in ['full', 'reduced'], f"Invalid mode: {mode}"
        self.current_mode = mode


class AudioAugmentor:
    """Audio data augmentation for contrastive learning."""
    
    def __init__(self, config: Dict):
        """Initialize augmentor with configuration."""
        self.config = config['edge']['augmentation']
        self.sample_rate = config['data']['sample_rate']
        
    def time_shift(self, waveform: torch.Tensor, max_shift: float = 0.2) -> torch.Tensor:
        """
        Apply random time shifting.
        
        Args:
            waveform: Input audio tensor
            max_shift: Maximum shift as fraction of length
            
        Returns:
            Shifted waveform
        """
        shift = int(np.random.uniform(-max_shift, max_shift) * waveform.shape[-1])
        return torch.roll(waveform, shift, dims=-1)
    
    def frequency_mask(self, mel_spec: torch.Tensor, mask_param: int = 10) -> torch.Tensor:
        """
        Apply frequency masking.
        
        Args:
            mel_spec: Mel-spectrogram tensor [batch, n_mels, time]
            mask_param: Maximum number of consecutive mel bins to mask
            
        Returns:
            Masked mel-spectrogram
        """
        n_mels = mel_spec.shape[1]
        mask_size = np.random.randint(0, mask_param)
        mask_start = np.random.randint(0, n_mels - mask_size) if mask_size < n_mels else 0
        
        mel_spec = mel_spec.clone()
        mel_spec[:, mask_start:mask_start + mask_size, :] = 0
        
        return mel_spec
    
    def amplitude_scale(self, waveform: torch.Tensor, 
                       scale_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """
        Apply random amplitude scaling.
        
        Args:
            waveform: Input audio tensor
            scale_range: Min and max scaling factors
            
        Returns:
            Scaled waveform
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return waveform * scale
    
    def augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentation pipeline.
        
        Args:
            waveform: Input audio tensor
            
        Returns:
            Augmented waveform
        """
        # Time shift
        if np.random.rand() < 0.5:
            waveform = self.time_shift(waveform, self.config['time_shift'])
        
        # Amplitude scaling
        if np.random.rand() < 0.5:
            waveform = self.amplitude_scale(waveform, self.config['amplitude_scale'])
        
        return waveform


class AdaptiveFeatureExtractor:
    """Adaptive feature extraction policy based on resource availability."""
    
    def __init__(self, config: Dict, resource_monitor):
        """
        Initialize adaptive policy.
        
        Args:
            config: Configuration dictionary
            resource_monitor: ResourceMonitor instance
        """
        self.config = config['edge']['adaptive_policy']
        self.resource_monitor = resource_monitor
        
        # Thresholds
        self.threshold = self.config['resource_threshold']
        self.hysteresis_up = self.config['hysteresis_up']
        self.hysteresis_down = self.config['hysteresis_down']
        
        # Weights for resource aggregation
        self.weights = self.config['weights']
        
        # Current mode
        self.current_mode = 'full'
        
    def get_resource_score(self) -> float:
        """
        Compute aggregated resource score.
        
        Returns:
            Resource score in [0, 1]
        """
        state = self.resource_monitor.get_state()
        
        score = (
            self.weights['cpu'] * state['cpu_util'] +
            self.weights['memory'] * state['mem_usage'] +
            self.weights['battery'] * (1.0 - state['battery_level']) +
            self.weights['thermal'] * state['thermal_throttling']
        )
        
        return score
    
    def should_switch_mode(self) -> Optional[str]:
        """
        Determine if processing mode should change.
        
        Returns:
            New mode ('full' or 'reduced') or None if no change
        """
        if not self.config['enabled']:
            return None
            
        resource_score = self.get_resource_score()
        
        # Add hysteresis to prevent oscillation
        if self.current_mode == 'full':
            # Switch to reduced if resources are constrained
            if resource_score >= self.threshold:
                return 'reduced'
        else:  # current_mode == 'reduced'
            # Switch back to full if resources are available
            if resource_score <= (self.threshold - self.hysteresis_down):
                return 'full'
        
        return None
    
    def update_mode(self) -> str:
        """
        Update and return current processing mode.
        
        Returns:
            Current processing mode
        """
        new_mode = self.should_switch_mode()
        if new_mode is not None:
            self.current_mode = new_mode
            
        return self.current_mode
