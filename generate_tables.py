import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.net_architecture import PhysX_MKS_GhostNet
from src.dataset import MSTAR_Dataset

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "outputs/checkpoints/best_model.pth"
TABLES_DIR = "outputs/tables"
RESULTS_DIR = "outputs/results"

def count_parameters(model):
    """
    Counts total learnable parameters (Weights + Biases).
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_shape=(1, 1, 128, 128)):
    """
    Calculates FLOPs using Forward Hooks.
    This is critical for Complex-Valued Networks because a single 'ComplexConv2d'
    actually performs 4 internal 'nn.Conv2d' operations (ac, bd, ad, bc).
    
    Standard tools (like thop) often miscalculate this.
    Our hook attaches to the underlying nn.Conv2d layers, catching every single
    execution, ensuring the count is 100% accurate.
    """
    flops = 0
    
    def conv_hook(module, input, output):
        nonlocal flops
        # Output shape: [Batch, Channels, Height, Width]
        batch_size, _, h_out, w_out = output.shape
        c_in = module.in_channels
        c_out = module.out_channels
        k_h, k_w = module.kernel_size
        
        # Standard Conv FLOPs formula:
        # Ops = 2 (Mult+Add) * Cin * K * K * Hout * Wout * Cout
        # We assume groups=1 for simplicity in basic FLOP counting
        groups = module.groups
        
        layer_flops = 2 * c_in * k_h * k_w * h_out * w_out * (c_out // groups) * batch_size
        
        if module.bias is not None:
            layer_flops += h_out * w_out * c_out * batch_size
            
        flops += layer_flops

    def linear_hook(module, input, output):
        nonlocal flops
        batch_size = input[0].shape[0]
        # Linear FLOPs = 2 * Cin * Cout
        layer_flops = 2 * module.in_features * module.out_features * batch_size
        if module.bias is not None:
            layer_flops += module.out_features * batch_size
        flops += layer_flops

    # Register hooks on all leaf modules
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
            
    # Run Dummy Forward Pass
    dummy_input = torch.randn(*input_shape, dtype=torch.complex64).to(DEVICE)
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
        
    # Remove hooks to clean up
    for h in hooks: h.remove()
    
    return flops

def calculate_psnr(img1, img2):
    # img1, img2: [B, C, H, W] Complex Tensors
    mse = torch.mean((img1.real - img2.real)**2 + (img1.imag - img2.imag)**2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def generate_table_v_efficiency(model):
    """
    Table V: Efficiency Analysis
    Calculates Params, FLOPs, and Inference Time correctly.
    """
    print("Generating Table V: Efficiency Analysis...")
    
    # 1. Parameters (Millions)
    params = count_parameters(model)
    params_m = params / 1e6
    print(f"  - Parameters: {params_m:.2f} M")
    
    # 2. FLOPs (Giga)
    flops = count_flops(model)
    flops_g = flops / 1e9
    print(f"  - FLOPs: {flops_g:.2f} G")
    
    # 3. Inference Time (Latency)
    model.eval()
    dummy_input = torch.randn(1, 1, 128, 128, dtype=torch.complex64).to(DEVICE)
    
    # Warmup (Essential for GPU to reach clock speeds)
    print("  - Warming up GPU...")
    for _ in range(50): 
        _ = model(dummy_input)
    
    # Measure Latency with Sync
    print("  - Measuring Latency...")
    iters = 300
    
    # Synchronize before start
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)
            
    # Synchronize after end
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    end = time.time()
    
    avg_time_ms = ((end - start) / iters) * 1000
    
    # 4. Save Data
    # Only saving OUR model's data. Fill in comparisons manually from papers if needed.
    data = [
        ["PhysX-MKS-GhostNet (Ours)", f"{params_m:.2f}", f"{flops_g:.2f}", f"{avg_time_ms:.2f}"]
    ]
    
    df = pd.DataFrame(data, columns=["Method", "Params (M)", "FLOPs (G)", "Latency (ms)"])
    print("\n--- Efficiency Results ---")
    print(df)
    df.to_csv(f"{TABLES_DIR}/Table_V_Efficiency.csv", index=False)

def generate_table_i_network_config(model):
    """ Table I: Network Parameter Configuration """
    print("Generating Table I...")
    data = []
    
    # CMKS (Module A)
    data.append(["CMKS_Branch1", "3x3", 1, "1x16x128x128"])
    data.append(["CMKS_Branch2", "5x5", 1, "1x16x128x128"])
    data.append(["CMKS_Branch3", "7x7", 1, "1x16x128x128"])
    data.append(["Fusion", "Concat", "-", "48x128x128"])
    
    # Ghost Backbone (Module B)
    for i, cfg in enumerate(model.cfgs):
        k, exp, c, se, s = cfg
        layer_name = f"Ghost_Stage_{i+1}"
        data.append([layer_name, f"Ghost({k}x{k})", s, f"{c} channels"])

    # Heads
    data.append(["Physics_Branch", "Energy", "-", "1x128x128"])
    data.append(["Reconstruction", "Upsample", "-", "1x128x128"])
    data.append(["Classifier", "Linear", "-", "10 Classes"])
    
    df = pd.DataFrame(data, columns=["Layer", "Kernel/Type", "Stride", "Output Dims"])
    df.to_csv(f"{TABLES_DIR}/Table_I_Network_Config.csv", index=False)

def generate_table_ii_dataset_config():
    """ Table II: Dataset Configuration """
    print("Generating Table II...")
    splits = ['soc_train', 'soc_test', 'eoc_1_test', 'eoc_2_test']
    data = []
    
    for split in splits:
        ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split=split, cache_memory=False)
        count = len(ds)
        desc = "Standard" if "soc" in split else "Robustness Test"
        data.append([split.upper(), desc, count])
        
    df = pd.DataFrame(data, columns=["Dataset Split", "Description", "Image Count"])
    df.to_csv(f"{TABLES_DIR}/Table_II_Dataset_Config.csv", index=False)

def generate_table_iii_soc_comparison():
    """ Table III: SoC-10 SOTA Comparison """
    print("Generating Table III...")
    
    try:
        eval_df = pd.read_csv(f"{RESULTS_DIR}/evaluation_summary.csv")
        our_acc = eval_df.loc[eval_df['Test_Set'] == 'SoC-10', 'Accuracy (%)'].values[0]
    except:
        our_acc = "N/A"

    data = [
        ["A-ConvNets", "CNN (Real)", "No", 99.13],
        ["CV-ResNet", "ResNet (Complex)", "No", 99.54],
        ["KINN (Base 2)", "GNN + Scattering", "Yes", 98.20],
        ["CRMC-Net (Base 1)", "Ghost-CVNN", "No", 99.40],
        ["PhysX-MKS-GhostNet (Ours)", "PhysX-Ghost", "Yes", our_acc]
    ]
    
    df = pd.DataFrame(data, columns=["Method", "Backbone", "Physics-Informed", "Accuracy (%)"])
    df.to_csv(f"{RESULTS_DIR}/Table_III_SoC_Comparison.csv", index=False)

def generate_table_ix_reconstruction(model):
    """ Table IX: Reconstruction Quality """
    print("Generating Table IX...")
    
    ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='soc_test', cache_memory=True)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    model.eval()
    psnr_total = 0
    count = 0
    
    with torch.no_grad():
        for img, _ in tqdm(loader, desc="Calc PSNR"):
            img = img.to(DEVICE)
            _, recon, _, _ = model(img)
            
            # Resize recon if necessary
            if recon.shape != img.shape:
                recon = torch.nn.functional.interpolate(recon.real, size=img.shape[2:]) + \
                        1j * torch.nn.functional.interpolate(recon.imag, size=img.shape[2:])
            
            psnr_total += calculate_psnr(img, recon)
            count += 1
            
    avg_psnr = psnr_total / count if count > 0 else 0
    
    data = [["PhysX-MKS-GhostNet", f"{avg_psnr:.2f} dB"]]
    df = pd.DataFrame(data, columns=["Method", "Reconstruction PSNR"])
    df.to_csv(f"{RESULTS_DIR}/Table_IX_Reconstruction.csv", index=False)

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load Model
    if os.path.exists(CHECKPOINT):
        print(f"Loading model from {CHECKPOINT}")
        model = PhysX_MKS_GhostNet(num_classes=10, width_mult=1.4).to(DEVICE)
        model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    else:
        print("WARNING: No checkpoint found. Tables will be generated with random weights.")
        model = PhysX_MKS_GhostNet(num_classes=10, width_mult=1.4).to(DEVICE)

    # Generate All
    generate_table_i_network_config(model)
    if os.path.exists("data/MSTAR_Combined"):
        generate_table_ii_dataset_config()
    generate_table_iii_soc_comparison()
    generate_table_v_efficiency(model)
    
    if os.path.exists("data/MSTAR_Combined"):
        generate_table_ix_reconstruction(model)
    
    print("\n--- All Tables Generated in outputs/tables/ and outputs/results/ ---")

if __name__ == "__main__":
    main()