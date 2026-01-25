import torch
import torch.nn as nn
import math
from .complex_layers import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU

class ComplexGhostModule(nn.Module):
    """
    Complex-Valued Ghost Module.
    Splits generation into:
    1. Intrinsic Maps (Heavy Complex Conv)
    2. Ghost Maps (Cheap Depthwise Complex Conv)
    
    Paper Reference: "GhostNet: More Features from Cheap Operations" (Adapted to Complex Domain)
    """
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(ComplexGhostModule, self).__init__()
        self.oup = oup
        
        # Calculate channels for split
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # 1. Intrinsic (Heavy) Complex Convolution
        # Generates the "seed" feature maps with full computation
        self.primary_conv = nn.Sequential(
            ComplexConv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            ComplexBatchNorm2d(init_channels),
            ComplexReLU() if relu else nn.Sequential()
        )

        # 2. Cheap (Linear) Complex Operation -> Depthwise Conv
        # Generates "ghost" maps from the seed maps efficiently
        # Groups = init_channels makes this a Depthwise Convolution
        self.cheap_operation = nn.Sequential(
            ComplexConv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            ComplexBatchNorm2d(new_channels),
            ComplexReLU() if relu else nn.Sequential()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class ComplexGhostBottleneck(nn.Module):
    """
    The main building block of PhysX-MKS-GhostNet.
    Structure:
    1. Ghost Module (Expansion)
    2. Depthwise Conv (if stride=2)
    3. Ghost Module (Reduction)
    4. Residual Connection
    """
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, se_ratio=0):
        super(ComplexGhostBottleneck, self).__init__()
        self.stride = stride

        # 1. Expansion Phase (Increase dimensions)
        self.ghost1 = ComplexGhostModule(in_chs, mid_chs, relu=True)
        
        # 2. Depthwise Convolution (Spatial Downsampling)
        # Only applied if we are reducing spatial size (stride=2)
        if self.stride > 1:
            self.conv_dw = ComplexConv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride, 
                                         padding=(dw_kernel_size-1)//2, groups=mid_chs, bias=False)
            self.bn_dw = ComplexBatchNorm2d(mid_chs)

        # 3. Reduction Phase (Project back to output dimensions)
        self.ghost2 = ComplexGhostModule(mid_chs, out_chs, relu=False)
        
        # 4. Shortcut Connection (Residual)
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            # If dimensions/stride mismatch, use a 1x1 conv to align them
            self.shortcut = nn.Sequential(
                ComplexConv2d(in_chs, in_chs, dw_kernel_size, stride=stride, 
                              padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                ComplexBatchNorm2d(in_chs),
                ComplexConv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                ComplexBatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        # Main Path
        x1 = self.ghost1(x)
        
        if self.stride > 1:
            x1 = self.conv_dw(x1)
            x1 = self.bn_dw(x1)

        x1 = self.ghost2(x1)
        
        # Element-wise complex addition
        return x1 + residual