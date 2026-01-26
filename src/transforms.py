import torch
import numpy as np
import torchvision.transforms.functional as TF
import random

class RandomPhaseShift(object):
    """
    Operation: Z' = Z * e^(j * theta)
    Forces model to learn relative phase structures.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if np.random.random() < self.p:
            theta = (torch.rand(1) * 2 * np.pi) - np.pi
            phase_rot = torch.polar(torch.ones(1), theta).to(x.device)
            return x * phase_rot
        return x

class ComplexRandomRotation(object):
    """
    True Spatial Rotation for Complex Tensors.
    Rotates real and imaginary parts identically.
    """
    def __init__(self, degrees=15, p=0.5):
        self.degrees = degrees # Now actually used!
        self.p = p

    def __call__(self, x):
        # x: [C, H, W] (Complex64)
        if np.random.random() < self.p:
            # Get random angle
            angle = random.uniform(-self.degrees, self.degrees)
            
            # 1. Stack Real/Imag into a [2, H, W] or [2*C, H, W] block
            # For 1-channel complex input (1, H, W):
            # We treat it as 2 channels (Real, Imag) to rotate them together
            real = x.real
            imag = x.imag
            stacked = torch.cat([real, imag], dim=0) # [2*C, H, W]
            
            # 2. Rotate using efficient bilinear interpolation
            # TF.rotate expects [..., H, W]
            rotated_stacked = TF.rotate(stacked, angle, interpolation=TF.InterpolationMode.BILINEAR)
            
            # 3. Unstack and reform complex tensor
            channels = x.shape[0]
            rot_real = rotated_stacked[:channels, ...]
            rot_imag = rotated_stacked[channels:, ...]
            
            return rot_real + 1j * rot_imag
        return x

class ComplexGaussianNoise(object):
    """
    Simulates Thermal Noise / Speckle.
    """
    def __init__(self, sigma=0.05, p=0.2):
        self.sigma = sigma
        self.p = p
    
    def __call__(self, x):
        if np.random.random() < self.p:
            noise_r = torch.randn_like(x.real) * self.sigma
            noise_i = torch.randn_like(x.imag) * self.sigma
            return x + (noise_r + 1j * noise_i)
        return x