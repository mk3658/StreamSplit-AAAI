"""
MobileNetV3-Small encoder for edge devices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ConvBNActivation(nn.Sequential):
    """Convolution + BatchNorm + Activation block."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, activation=nn.ReLU):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                     padding, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels, squeeze_factor=4):
        super().__init__()
        squeeze_channels = max(1, channels // squeeze_factor)
        self.fc1 = nn.Conv2d(channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, channels, 1)
        
    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class InvertedResidual(nn.Module):
    """Inverted residual block for MobileNetV3."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio, use_se=False, activation=nn.ReLU):
        super().__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.append(ConvBNActivation(in_channels, hidden_dim, 
                                          kernel_size=1, 
                                          activation=activation))
        
        # Depthwise
        layers.append(ConvBNActivation(hidden_dim, hidden_dim, 
                                      kernel_size=kernel_size,
                                      stride=stride, 
                                      activation=activation))
        
        # Squeeze-and-Excitation
        if use_se:
            layers.append(SqueezeExcitation(hidden_dim))
        
        # Project
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small for audio embeddings."""
    
    def __init__(self, width_mult=0.75, embedding_dim=128, 
                 input_channels=1, use_early_exit=True):
        super().__init__()
        
        self.width_mult = width_mult
        self.embedding_dim = embedding_dim
        self.use_early_exit = use_early_exit
        
        # Helper function to scale channels
        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        # First conv
        input_channel = _make_divisible(16 * width_mult)
        self.features = nn.ModuleList([
            ConvBNActivation(input_channels, input_channel, 
                           kernel_size=3, stride=2, 
                           activation=nn.Hardswish)
        ])
        
        # Inverted residual blocks configuration
        # [kernel, exp_ratio, out_ch, se, activation, stride]
        configs = [
            [3, 1, 16, True, nn.ReLU, 2],
            [3, 4.5, 24, False, nn.ReLU, 2],
            [3, 3.67, 24, False, nn.ReLU, 1],
            [5, 4, 40, True, nn.Hardswish, 2],
            [5, 6, 40, True, nn.Hardswish, 1],
            [5, 6, 40, True, nn.Hardswish, 1],
            [5, 3, 48, True, nn.Hardswish, 1],
            [5, 3, 48, True, nn.Hardswish, 1],
            [5, 6, 96, True, nn.Hardswish, 2],
            [5, 6, 96, True, nn.Hardswish, 1],
            [5, 6, 96, True, nn.Hardswish, 1],
        ]
        
        # Build inverted residual blocks
        for k, exp, c, se, act, s in configs:
            output_channel = _make_divisible(c * width_mult)
            self.features.append(
                InvertedResidual(input_channel, output_channel, 
                               k, s, exp, se, act)
            )
            input_channel = output_channel
        
        # Final conv
        final_conv_ch = _make_divisible(576 * width_mult)
        self.features.append(
            ConvBNActivation(input_channel, final_conv_ch, 
                           kernel_size=1, activation=nn.Hardswish)
        )
        
        # Pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(final_conv_ch, embedding_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Early exit point (after first few blocks)
        self.early_exit_index = 6 if use_early_exit else None
        if use_early_exit:
            early_exit_ch = _make_divisible(40 * width_mult)
            self.early_exit_classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(early_exit_ch, embedding_dim)
            )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, use_early_exit=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, channels, height, width]
            use_early_exit: Whether to use early exit
            
        Returns:
            Embeddings [batch, embedding_dim]
        """
        # Process through feature layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            
            # Early exit if requested
            if (use_early_exit and self.use_early_exit and 
                i == self.early_exit_index):
                return self.early_exit_classifier(x)
        
        # Full forward pass
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def get_intermediate_features(self, x, layer_index):
        """Get features from intermediate layer."""
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == layer_index:
                return x
        return x
