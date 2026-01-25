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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_psnr(img1, img2):
    # img1, img2: [B, C, H, W] Complex Tensors
    mse = torch.mean((img1.real - img2.real)**2 + (img1.imag - img2.imag)**2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def generate_table_i_network_config(model):
    """
    Table I: Network Parameter Configuration
    Lists the specific layer configurations (Kernel, Stride, Input/Output).
    """
    print("Generating Table I...")
    data = []
    
    # Manually extracting layer info based on the architecture definition
    # This reflects the specific design of PhysX-MKS-GhostNet
    
    # 1. CMKS (Module A)
    data.append(["CMKS_Branch1", "3x3", 1, "1x16x128x128"])
    data.append(["CMKS_Branch2", "5x5", 1, "1x16x128x128"])
    data.append(["CMKS_Branch3", "7x7", 1, "1x16x128x128"])
    data.append(["Fusion", "Concat", "-", "48x128x128"])
    
    # 2. Ghost Backbone (Module B)
    # We iterate through the defined configs in the model
    for i, cfg in enumerate(model.cfgs):
        k, exp, c, se, s = cfg
        layer_name = f"Ghost_Stage_{i+1}"
        data.append([layer_name, f"Ghost({k}x{k})", s, f"{c} channels"])

    # 3. Heads
    data.append(["Physics_Branch", "Energy", "-", "1x128x128"])
    data.append(["Reconstruction", "Upsample", "-", "1x128x128"])
    data.append(["Classifier", "Linear", "-", "10 Classes"])
    
    df = pd.DataFrame(data, columns=["Layer", "Kernel/Type", "Stride", "Output Dims"])
    df.to_csv(f"{TABLES_DIR}/Table_I_Network_Config.csv", index=False)

def generate_table_ii_dataset_config():
    """
    Table II: Dataset Configuration
    Counts exact images in each split.
    """
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
    """
    Table III: SoC-10 SOTA Comparison
    Merges our results with Base Papers.
    """
    print("Generating Table III...")
    
    # 1. Get Our Accuracy
    try:
        eval_df = pd.read_csv(f"{RESULTS_DIR}/evaluation_summary.csv")
        our_acc = eval_df.loc[eval_df['Test_Set'] == 'SoC-10', 'Accuracy (%)'].values[0]
    except:
        our_acc = "N/A (Run evaluate.py)"

    # 2. Hardcoded SOTA (From Project Overview / Literature)
    data = [
        ["A-ConvNets", "CNN (Real)", "No", 99.13],
        ["CV-ResNet", "ResNet (Complex)", "No", 99.54],
        ["KINN (Base 2)", "GNN + Scattering", "Yes", 98.20],
        ["CRMC-Net (Base 1)", "Ghost-CVNN", "No", 99.40], # Approx from paper
        ["PhysX-MKS-GhostNet (Ours)", "PhysX-Ghost", "Yes", our_acc]
    ]
    
    df = pd.DataFrame(data, columns=["Method", "Backbone", "Physics-Informed", "Accuracy (%)"])
    df.to_csv(f"{RESULTS_DIR}/Table_III_SoC_Comparison.csv", index=False)

def generate_table_v_efficiency(model):
    """
    Table V: Efficiency Analysis
    Calculates Params, FLOPs (Estimate), and Inference Time.
    """
    print("Generating Table V...")
    
    # 1. Params
    params_m = count_parameters(model) / 1e6
    
    # 2. Inference Time
    model.eval()
    dummy_input = torch.randn(1, 1, 128, 128, dtype=torch.complex64).to(DEVICE)
    
    # Warmup
    for _ in range(10): _ = model(dummy_input)
    
    # Measure
    start = time.time()
    iters = 100
    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)
    end = time.time()
    
    avg_time_ms = ((end - start) / iters) * 1000
    
    # 3. Comparison Data
    data = [
        ["MobileNetV3 (Real)", 5.4, 21.0],
        ["CRMC-Net (Base 1)", 0.8, 45.0], # CRMC is heavy on complex convs
        ["PhysX-MKS-GhostNet (Ours)", f"{params_m:.2f}", f"{avg_time_ms:.2f}"]
    ]
    
    df = pd.DataFrame(data, columns=["Method", "Params (M)", "Inference Time (ms)"])
    df.to_csv(f"{TABLES_DIR}/Table_V_Efficiency.csv", index=False)

def generate_table_ix_reconstruction(model):
    """
    Table IX: Reconstruction Quality
    Computes PSNR on SoC-Test set.
    """
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
        print("WARNING: No checkpoint found. Generating tables with initialized weights (random).")
        model = PhysX_MKS_GhostNet(num_classes=10, width_mult=1.4).to(DEVICE)

    # Generate All
    generate_table_i_network_config(model)
    # Note: Table II requires data directory to be present
    if os.path.exists("data/MSTAR_Combined"):
        generate_table_ii_dataset_config()
    generate_table_iii_soc_comparison()
    generate_table_v_efficiency(model)
    
    # Table IV is already generated by evaluate.py (evaluation_summary.csv)
    
    if os.path.exists("data/MSTAR_Combined"):
        generate_table_ix_reconstruction(model)
    
    print("\n--- All Tables Generated in outputs/tables/ and outputs/results/ ---")

if __name__ == "__main__":
    main()