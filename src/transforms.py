import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
import math

class RandomPhaseShift(object):
    """
    Operation: Z' = Z * e^(j * theta)
    Forces model to ignore absolute phase, learning relative phase structures.
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
    """
    def __init__(self, degrees=15, p=0.5):
        self.degrees = degrees 
        self.p = p

    def __call__(self, x):
        if np.random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            
            # Stack Real/Imag for joint rotation
            real = x.real
            imag = x.imag
            stacked = torch.cat([real, imag], dim=0) 
            
            # Rotate
            rotated_stacked = TF.rotate(stacked, angle, interpolation=TF.InterpolationMode.BILINEAR)
            
            # Unstack
            channels = x.shape[0]
            rot_real = rotated_stacked[:channels, ...]
            rot_imag = rotated_stacked[channels:, ...]
            
            return rot_real + 1j * rot_imag
        return x

class ComplexRandomScale(object):
    """
    Simulates Depression Angle Variations (EoC-1).
    Aggressive scaling (0.7 - 1.3) forces robustness.
    """
    def __init__(self, scale_range=(0.7, 1.3), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, x):
        if np.random.random() < self.p:
            scale = random.uniform(*self.scale_range)
            
            real = x.real
            imag = x.imag
            stacked = torch.cat([real, imag], dim=0)
            
            _, h, w = stacked.shape
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # Resize
            scaled = TF.resize(stacked, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            
            # Pad or Crop back to original size
            if scale < 1.0:
                pad_h = max(0, h - new_h)
                pad_w = max(0, w - new_w)
                pad_l = pad_w // 2
                pad_r = pad_w - pad_l
                pad_t = pad_h // 2
                pad_b = pad_h - pad_t
                scaled = TF.pad(scaled, [pad_l, pad_t, pad_r, pad_b], fill=0)
            
            scaled = TF.center_crop(scaled, [h, w])
            
            c = x.shape[0]
            return scaled[:c] + 1j * scaled[c:]
        return x

class ComplexSpeckleNoise(object):
    """
    Multiplicative Noise (SAR Speckle).
    High sigma (0.2) simulates poor operating conditions.
    """
    def __init__(self, prob=0.5, sigma=0.2):
        self.prob = prob
        self.sigma = sigma

    def __call__(self, x):
        if np.random.random() < self.prob:
            # Multiplicative noise: Z' = Z * (1 + N(0, sigma))
            noise = torch.randn_like(x.real) * self.sigma
            return x * (1 + noise)
        return x

class ComplexRandomErasing(object):
    """
    [NEW] Randomly erases a rectangle in the complex image.
    Simulates occlusion or missing parts (Variant Robustness).
    """
    def __init__(self, p=0.5, scale=(0.02, 0.20), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, x):
        if np.random.random() < self.p:
            img_c, img_h, img_w = x.shape
            area = img_h * img_w

            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_w and h < img_h:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)

                # Erase (Set Real and Imag to 0)
                mask = torch.ones_like(x.real)
                mask[:, i:i+h, j:j+w] = 0
                return x * mask
        return x