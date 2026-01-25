import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexConv2d(nn.Module):
    """
    Implements Complex-Valued Convolution.
    Performs the operation: (Re + j*Im) * (K_Re + j*K_Im)
    Output Re = (Re * K_Re) - (Im * K_K_Im)
    Output Im = (Re * K_Im) + (Im * K_Re)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        # We need separate filters for Real and Imaginary components
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

    def forward(self, x):
        # x is a ComplexTensor (real, imag) or standard torch.complex
        if torch.is_complex(x):
            xr, xi = x.real, x.imag
        else:
            # Fallback if passed as tuple/list (legacy support)
            xr, xi = x[0], x[1]
            
        # (ac - bd)
        real_part = self.conv_r(xr) - self.conv_i(xi)
        # (ad + bc)
        imag_part = self.conv_r(xi) + self.conv_i(xr)
        
        return torch.complex(real_part, imag_part)

class ComplexBatchNorm2d(nn.Module):
    """
    Magnitude-based Batch Normalization.
    Normalizes by the shared magnitude variance (E[|x|^2]) to PRESERVE PHASE information.
    Standard BN on Real/Imag separately would destroy the relative phase angles.
    
    Paper Reference: "A Multiscale Convolution SAR... Based on Complex-Valued Neural Networks"
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(ComplexBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable affine parameters (Gamma, Beta)
        # Gamma scales the magnitude, Beta shifts the center
        self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_(0.9, 1.1))
        self.beta_r = nn.Parameter(torch.zeros(num_features))
        self.beta_i = nn.Parameter(torch.zeros(num_features))
        
        # Running stats
        self.register_buffer('running_mean_r', torch.zeros(num_features))
        self.register_buffer('running_mean_i', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features)) # Magnitude Variance

    def forward(self, x):
        xr, xi = x.real, x.imag
        
        if self.training:
            # 1. Calculate Spatial Means
            mu_r = xr.mean(dim=[0, 2, 3])
            mu_i = xi.mean(dim=[0, 2, 3])
            
            # 2. Center the data
            xr_cent = xr - mu_r[None, :, None, None]
            xi_cent = xi - mu_i[None, :, None, None]
            
            # 3. Calculate Combined Variance (V = E[|x|^2])
            # This is the "Magnitude Variance" - KEY for Phase Preservation
            var = (xr_cent**2 + xi_cent**2).mean(dim=[0, 2, 3])
            
            # 4. Update Running Stats
            with torch.no_grad():
                self.running_mean_r = (1 - self.momentum) * self.running_mean_r + self.momentum * mu_r
                self.running_mean_i = (1 - self.momentum) * self.running_mean_i + self.momentum * mu_i
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # Inference Mode
            mu_r = self.running_mean_r
            mu_i = self.running_mean_i
            var = self.running_var
            
            xr_cent = xr - mu_r[None, :, None, None]
            xi_cent = xi - mu_i[None, :, None, None]

        # 5. Normalize
        std = torch.sqrt(var + self.eps)
        xr_norm = xr_cent / std[None, :, None, None]
        xi_norm = xi_cent / std[None, :, None, None]
        
        # 6. Affine Transform
        xr_out = (self.gamma[None, :, None, None] * xr_norm) + self.beta_r[None, :, None, None]
        xi_out = (self.gamma[None, :, None, None] * xi_norm) + self.beta_i[None, :, None, None]
        
        return torch.complex(xr_out, xi_out)

class ComplexReLU(nn.Module):
    """
    Complex ReLU (CReLU).
    Applies ReLU to Real and Imaginary parts separately.
    Result lies in the first quadrant of the complex plane.
    """
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))

class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(ComplexAvgPool2d, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return torch.complex(self.pool(x.real), self.pool(x.imag))

class ComplexUpsample(nn.Module):
    """
    Custom Upsampling for Complex Tensors.
    Splits Real/Imag, interpolates separately, and recombines.
    Crucial for the Reconstruction Decoder (Module E).
    """
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=True):
        super(ComplexUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        # x is Complex (B, C, H, W)
        xr, xi = x.real, x.imag
        
        # Upsample separately
        xr_up = F.interpolate(xr, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        xi_up = F.interpolate(xi, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        
        return torch.complex(xr_up, xi_up)