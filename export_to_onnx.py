import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import types

# Ensure src is in path
sys.path.append(os.getcwd())

# --- CONFIG ---
DEVICE = torch.device("cpu")
CHECKPOINT_PATH = "outputs/checkpoints/best_model.pth"
ONNX_PATH = "outputs/physx_ghost.onnx"

# =========================================================================
# 1. SHADOW LAYERS (Real-Valued Equivalents)
# =========================================================================

class RealComplexConv2d(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.conv_r = original_layer.conv_r
        self.conv_i = original_layer.conv_i
        self.groups = original_layer.conv_r.groups
        
    def forward(self, x):
        # Input is (B, 2C, H, W) -> Split into Real/Imag
        channels = x.shape[1] // 2
        xr = x[:, :channels, :, :]
        xi = x[:, channels:, :, :]
        
        real_part = self.conv_r(xr) - self.conv_i(xi)
        imag_part = self.conv_r(xi) + self.conv_i(xr)
        
        return torch.cat([real_part, imag_part], dim=1)

class RealComplexBatchNorm2d(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.layer = original_layer
        
    def forward(self, x):
        channels = x.shape[1] // 2
        xr = x[:, :channels, :, :]
        xi = x[:, channels:, :, :]
        
        mu_r, mu_i = self.layer.running_mean_r, self.layer.running_mean_i
        var = self.layer.running_var
        eps = self.layer.eps
        gamma = self.layer.gamma
        beta_r, beta_i = self.layer.beta_r, self.layer.beta_i
        
        std = torch.sqrt(var + eps)
        xr_norm = (xr - mu_r[None, :, None, None]) / std[None, :, None, None]
        xi_norm = (xi - mu_i[None, :, None, None]) / std[None, :, None, None]
        
        xr_out = (gamma[None, :, None, None] * xr_norm) + beta_r[None, :, None, None]
        xi_out = (gamma[None, :, None, None] * xi_norm) + beta_i[None, :, None, None]
        
        return torch.cat([xr_out, xi_out], dim=1)

class RealComplexReLU(nn.Module):
    def __init__(self, original_layer=None):
        super().__init__()
    def forward(self, x):
        return F.relu(x)

class RealComplexSimAM(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.e_lambda = original_layer.e_lambda
        self.activation = original_layer.activation

    def forward(self, x):
        channels = x.shape[1] // 2
        xr = x[:, :channels, :, :]
        xi = x[:, channels:, :, :]
        
        # Calculate Magnitude (Energy)
        mag = torch.sqrt(xr**2 + xi**2 + 1e-8)
        
        # SimAM Logic
        b, c, h, w = mag.size()
        n = h * w - 1
        x_minus_mu_square = (mag - mag.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        # Apply Activation
        att = self.activation(y)
        
        # Concatenate attention map with itself to match input size (2C)
        att_doubled = torch.cat([att, att], dim=1)
        
        return x * att_doubled

# --- Ghost Bottleneck Wrapper with Intelligent Resizing ---
class RealComplexGhostBottleneck(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        
        # 1. Patch Ghost Modules
        self.ghost1 = original_layer.ghost1
        recursive_patch(self.ghost1)
        
        self.ghost2 = original_layer.ghost2
        recursive_patch(self.ghost2)

        # 2. Patch Depthwise Conv
        if hasattr(original_layer, 'conv_dw') and original_layer.conv_dw is not None:
            self.conv_dw = RealComplexConv2d(original_layer.conv_dw)
            self.bn_dw = RealComplexBatchNorm2d(original_layer.bn_dw)
        else:
            self.conv_dw = None
            self.bn_dw = None

        # 3. Patch Shortcut
        self.shortcut = original_layer.shortcut
        if isinstance(self.shortcut, nn.Sequential):
            recursive_patch(self.shortcut)

    def forward(self, x):
        def run_ghost_module(module, inp):
            x1 = module.primary_conv(inp)
            x2 = module.cheap_operation(x1)
            return torch.cat([x1, x2], dim=1)
            
        # 1. Expansion
        hidden = run_ghost_module(self.ghost1, x)
        
        # --- INTELLIGENT RESIZING (The Fix) ---
        if self.conv_dw:
            expected_groups = self.conv_dw.groups
            expected_channels = expected_groups * 2 # Real + Imag
            current_channels = hidden.shape[1]
            
            # Case A: Too Few Channels -> Pad
            if current_channels < expected_channels:
                diff = expected_channels - current_channels
                padding = torch.zeros(hidden.shape[0], diff, hidden.shape[2], hidden.shape[3], device=hidden.device)
                hidden = torch.cat([hidden, padding], dim=1)
            
            # Case B: Too Many Channels -> Slice
            elif current_channels > expected_channels:
                # We need to slice carefully: Keep Top-Real and Top-Imag
                # Current: [Real_Full, Imag_Full]
                # Target: [Real_Trimmed, Imag_Trimmed]
                
                half_curr = current_channels // 2
                half_exp = expected_channels // 2
                
                real_part = hidden[:, :half_exp, :, :]
                imag_part = hidden[:, half_curr : half_curr+half_exp, :, :]
                
                hidden = torch.cat([real_part, imag_part], dim=1)
        
        # 2. Depthwise
        if self.conv_dw:
            hidden = self.conv_dw(hidden)
            hidden = self.bn_dw(hidden)
        
        # 3. Projection
        y = run_ghost_module(self.ghost2, hidden)
        
        # 4. Shortcut
        if self.shortcut:
            y += self.shortcut(x)
                
        return y

# =========================================================================
# 2. PATCHING UTILITIES
# =========================================================================

def recursive_patch(module):
    if module.__class__.__name__ == 'ComplexGhostBottleneck':
        return RealComplexGhostBottleneck(module)

    for name, child in module.named_children():
        type_name = child.__class__.__name__
        
        if type_name == 'ComplexGhostBottleneck':
            setattr(module, name, RealComplexGhostBottleneck(child))
        elif type_name == 'ComplexConv2d':
            setattr(module, name, RealComplexConv2d(child))
        elif type_name == 'ComplexBatchNorm2d':
            setattr(module, name, RealComplexBatchNorm2d(child))
        elif type_name == 'ComplexReLU':
            setattr(module, name, RealComplexReLU(child))
        elif type_name == 'ComplexSimAM':
            setattr(module, name, RealComplexSimAM(child))
        else:
            recursive_patch(child)
            
    return module

# =========================================================================
# 3. MAIN EXPORT LOGIC
# =========================================================================

def main():
    print("--- Exporting PhysX-MKS-GhostNet (Robust Shadow Patch) ---")
    
    from src.models.net_architecture import PhysX_MKS_GhostNet
    
    # 1. Load Model (Width=0.7)
    print("Loading PyTorch Checkpoint...")
    try:
        core_model = PhysX_MKS_GhostNet(num_classes=10, width_mult=0.7).to(DEVICE)
        core_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    # 2. Apply Patch
    print("Patching layers...")
    recursive_patch(core_model)
    
    # 3. Patch Network Forward
    def real_mode_forward(self, x):
        # CMKS
        b1, b2, b3 = self.cmks_branch1(x), self.cmks_branch2(x), self.cmks_branch3(x)
        x_fused = torch.cat([b1, b2, b3], dim=1)
        
        # Backbone
        features = self.ghost_stages(x_fused)
        features_att = self.simam(features)
        
        # Classifier
        x_pool = F.adaptive_avg_pool2d(features_att, 1)
        x_expand = self.conv_last(x_pool)
        x_flat = x_expand.view(x_expand.size(0), -1)
        class_logits = self.classifier(x_flat)
        
        return class_logits

    core_model.forward = types.MethodType(real_mode_forward, core_model)
    core_model.eval()

    # 4. Dummy Input (Real-View: 2 channels)
    dummy_input = torch.randn(1, 2, 128, 128, dtype=torch.float32).to(DEVICE)
    
    print(f"Exporting to {ONNX_PATH}...")
    torch.onnx.export(
        core_model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_2ch'],
        output_names=['logits'],
        dynamic_axes={'input_2ch': {0: 'batch_size'}}
    )
    print("✅ SUCCESS: ONNX model exported.")

if __name__ == "__main__":
    main()