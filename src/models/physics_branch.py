import torch
import torch.nn as nn
import torch.nn.functional as F
from .complex_layers import ComplexConv2d, ComplexReLU, ComplexUpsample

class PhysicsMapping(nn.Module):
    """
    Module D: Physics Scattering Branch (The "Brain")
    
    Purpose:
    Extracts geometric scattering centers (corners/edges) based on Feature Energy.
    This module forces the network to be "Physics-Informed" by explicitly 
    modeling where the electromagnetic energy is reflecting from.
    
    Paper Reference: "Knowledge-Informed Neural Network... (KINN)"
    """
    def __init__(self, in_channels=112, threshold_factor=0.8): 
        super(PhysicsMapping, self).__init__()
        self.threshold_factor = threshold_factor
        
        # Optional adapter if we want to learn the projection, 
        # but for pure physics, we often use direct magnitude.
        # We keep this 1x1 conv to allow the network to "weight" channels 
        # before energy extraction.
        self.adapter = ComplexConv2d(in_channels, 1, 1, 1, 0) 

    def forward(self, x):
        # Input: (B, C, H, W) Complex Features
        
        # 1. Calculate Feature Energy (Magnitude)
        # We average across channels to find spatial "hotspots"
        # |z| = sqrt(x^2 + y^2)
        mag = x.abs().mean(dim=1, keepdim=True) 
        
        # 2. Differentiable Peak Detection
        # We calculate a dynamic threshold based on the image's own energy level.
        # This makes it robust to signal strength variations (SoC vs EoC).
        # Global Average Pooling on spatial dims (2,3)
        threshold = mag.mean(dim=[2,3], keepdim=True) * self.threshold_factor
        
        # 3. Generate Scattering Map
        # ReLU(Energy - Threshold) keeps only the strong scattering centers
        # and zeros out the clutter/noise.
        scattering_map = F.relu(mag - threshold)
        
        # Normalize the map for stability (0 to 1)
        if scattering_map.max() > 0:
            scattering_map = scattering_map / (scattering_map.max() + 1e-6)
            
        return scattering_map

class ReconstructionDecoder(nn.Module):
    """
    Module E: Reconstruction Branch
    
    Purpose:
    Decodes high-level features back into a clean SAR image.
    This acts as a "Self-Supervised" regularizer. If the network can 
    reconstruct the tank, it 'understands' the tank's structure.
    
    Structure:
    Transposed Convolution equivalent using Upsample + Conv.
    """
    def __init__(self, in_channels=112):
        super(ReconstructionDecoder, self).__init__()
        
        # Upsample Block 1
        self.up1 = nn.Sequential(
            ComplexUpsample(scale_factor=2, mode='bilinear', align_corners=True),
            ComplexConv2d(in_channels, 64, 3, 1, 1),
            ComplexReLU()
        )
        
        # Upsample Block 2
        self.up2 = nn.Sequential(
            ComplexUpsample(scale_factor=2, mode='bilinear', align_corners=True),
            ComplexConv2d(64, 32, 3, 1, 1),
            ComplexReLU()
        )
        
        # Upsample Block 3
        self.up3 = nn.Sequential(
            ComplexUpsample(scale_factor=2, mode='bilinear', align_corners=True),
            ComplexConv2d(32, 16, 3, 1, 1),
            ComplexReLU()
        )
        
        # Final Projection to 1 Channel (Real+Imag)
        self.final = ComplexConv2d(16, 1, 3, 1, 1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.final(x)