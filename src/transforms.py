import torch
import numpy as np

class RandomPhaseShift(object):
    """
    PhysX Augmentation: Phase Shift.
    Operation: Z' = Z * e^(j * theta)
    
    Why: In SAR, the absolute phase is determined by the distance to the target.
    A shift of just a few millimeters changes the phase completely.
    This augmentation forces the model to ignore 'Absolute Phase' and learn 'Relative Phase'
    (the structural phase differences between pixels).
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        # x is (C, H, W) Complex Tensor
        if np.random.random() < self.p:
            # Generate random angle between -pi and pi
            theta = (torch.rand(1) * 2 * np.pi) - np.pi
            # Create phase rotation factor
            phase_rot = torch.polar(torch.ones(1), theta).to(x.device)
            return x * phase_rot
        return x

class ComplexRandomRotation(object):
    """
    Spatial Rotation for Complex Tensors.
    Rotates the H and W dimensions (last two dims).
    """
    def __init__(self, degrees=15, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, x):
        if np.random.random() < self.p:
            # Randomly rotate 90, 180, or 270 degrees
            # k is the number of times to rotate by 90 degrees
            k = np.random.randint(1, 4)
            # Rotate spatial dimensions [-2, -1]
            return torch.rot90(x, k, [-2, -1])
        return x

class ComplexGaussianNoise(object):
    """
    Simulates Thermal Noise / Speckle.
    Adds independent Gaussian noise to Real and Imaginary parts.
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