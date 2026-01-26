import torch
import torch.nn as nn
import torch.nn.functional as F
from .complex_layers import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU
from .ghost_module import ComplexGhostBottleneck
from .physics_branch import PhysicsMapping, ReconstructionDecoder

class ComplexSimAM(nn.Module):
    """
    Module C: Complex SimAM (Simple Attention Module)
    Parameter-free attention based on energy functions.
    """
    def __init__(self, e_lambda=1e-4):
        super(ComplexSimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        # Calculate energy on Magnitude
        mag = x.abs() 
        b, c, h, w = mag.size()
        n = h * w - 1
        
        # Calculate mean and variance of magnitude
        x_minus_mu_square = (mag - mag.mean(dim=[2, 3], keepdim=True)).pow(2)
        
        # Energy function
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        # Apply attention to both Real and Imaginary parts
        return x * self.activation(y)

class PhysX_MKS_GhostNet(nn.Module):
    """
    PhysX-MKS-GhostNet (Optimized for <0.3M Params)
    """
    def __init__(self, num_classes=10, width_mult=0.7, use_vlm=False, tiny=True):
        super(PhysX_MKS_GhostNet, self).__init__()
        self.use_vlm = use_vlm
        
        # --- CONFIGURATION FOR TINY MODE (<0.3M) ---
        # Optimized for strict <0.3M target while maintaining 99% acc potential.
        # Reduced head_c to 96 (was 128) and cmks_c to 10 (was 12).
        cmks_c = 10 if tiny else 16
        head_c = 96 if tiny else 256
        
        # --- Module A: Complex Multi-Kernel Scale (CMKS) Fusion ---
        self.cmks_branch1 = nn.Sequential(
            ComplexConv2d(1, cmks_c, 3, 1, 1), 
            ComplexBatchNorm2d(cmks_c), 
            ComplexReLU()
        )
        self.cmks_branch2 = nn.Sequential(
            ComplexConv2d(1, cmks_c, 5, 1, 2), 
            ComplexBatchNorm2d(cmks_c), 
            ComplexReLU()
        )
        self.cmks_branch3 = nn.Sequential(
            ComplexConv2d(1, cmks_c, 7, 1, 3), 
            ComplexBatchNorm2d(cmks_c), 
            ComplexReLU()
        )
        
        # --- Module B: Ghost Backbone ---
        # Configuration: [k, exp_size, c, se_ratio, stride]
        self.cfgs = [
            # Stage 1
            [3, 48, 24, 0, 2], 
            [3, 72, 24, 0, 1],
            # Stage 2
            [5, 120, 40, 0, 2], 
            [5, 120, 40, 0, 1],
            # Stage 3
            [3, 240, 80, 0, 2], 
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1], 
            [3, 480, 112, 0, 1],
        ]
        
        layers = []
        # Input channels = cmks_c * 3
        input_channel = cmks_c * 3
        
        # Calculate final channels based on width_mult
        final_channels = int(112 * width_mult)
        
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = int(c * width_mult)
            hidden_channel = int(exp_size * width_mult)
            layers.append(ComplexGhostBottleneck(input_channel, hidden_channel, output_channel, dw_kernel_size=k, stride=s))
            input_channel = output_channel
        self.ghost_stages = nn.Sequential(*layers)
        
        # --- Module C: Attention ---
        self.simam = ComplexSimAM()
        
        # --- Module D & E (Physics & Decoder) ---
        self.physics_branch = PhysicsMapping(in_channels=final_channels)
        self.decoder = ReconstructionDecoder(in_channels=final_channels)
        
        # --- Module F: VLM Projector (Optional) ---
        if self.use_vlm:
            self.vlm_projector = ComplexConv2d(final_channels, 768, 1, 1, 0)
        
        # --- Classifier ---
        self.conv_last = ComplexConv2d(final_channels, head_c, 1, 1, 0)
        
        # Final Linear Layer
        self.classifier = nn.Linear(head_c * 2, num_classes)

    def forward(self, x):
        # 1. CMKS Fusion
        b1 = self.cmks_branch1(x)
        b2 = self.cmks_branch2(x)
        b3 = self.cmks_branch3(x)
        x_fused = torch.cat([b1, b2, b3], dim=1)
        
        # 2. Backbone
        features = self.ghost_stages(x_fused)
        features_att = self.simam(features)
        
        # 3. Physics Tasks
        recon_img = self.decoder(features_att) 
        scatter_map = self.physics_branch(features_att) 
        
        # --- INJECT PHYSICS INTO CLASSIFIER ---
        features_phys = features_att * (1 + scatter_map)
        
        # 4. Vision-Language Projection (Optional)
        vlm_out = None
        if self.use_vlm:
            vlm_feat = self.vlm_projector(features_phys)
            vlm_pool = F.adaptive_avg_pool2d(vlm_feat.real, 1) + 1j * F.adaptive_avg_pool2d(vlm_feat.imag, 1)
            vlm_out = torch.cat([vlm_pool.real, vlm_pool.imag], dim=1).view(vlm_feat.size(0), -1)

        # 5. Classification
        x_pool = F.adaptive_avg_pool2d(features_phys.real, 1) + 1j * F.adaptive_avg_pool2d(features_phys.imag, 1)
        x_expand = self.conv_last(x_pool)
        x_flat = x_expand.view(x_expand.size(0), -1)
        
        # Concatenate Real and Imag parts for the final Linear Layer
        x_real_features = torch.cat([x_flat.real, x_flat.imag], dim=1)
        class_logits = self.classifier(x_real_features)
        
        return class_logits, recon_img, scatter_map, vlm_out