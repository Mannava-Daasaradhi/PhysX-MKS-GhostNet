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
from src.losses import PhysXLoss 

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "outputs/checkpoints/best_model.pth"
TABLES_DIR = "outputs/tables"
RESULTS_DIR = "outputs/results"

# --- HELPER: FLOP & PARAM COUNTER ---
def count_flops_params(model, input_shape=(1, 1, 128, 128)):
    flops = 0
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def conv_hook(module, input, output):
        nonlocal flops
        batch_size, _, h_out, w_out = output.shape
        c_in = module.in_channels
        c_out = module.out_channels
        k_h, k_w = module.kernel_size
        groups = module.groups
        layer_flops = 4 * c_in * k_h * k_w * h_out * w_out * (c_out // groups) * batch_size
        if module.bias is not None: layer_flops += h_out * w_out * c_out * batch_size
        flops += layer_flops

    def linear_hook(module, input, output):
        nonlocal flops
        batch_size = input[0].shape[0]
        layer_flops = 4 * module.in_features * module.out_features * batch_size
        flops += layer_flops

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d): hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear): hooks.append(m.register_forward_hook(linear_hook))
    
    dummy = torch.randn(*input_shape, dtype=torch.complex64).to(DEVICE)
    model.eval()
    with torch.no_grad(): _ = model(dummy)
    for h in hooks: h.remove()
    
    return params / 1e6, flops / 1e9

# --- HELPER: METRICS & LATEX ---
def get_metrics(model, split_name, filter_classes=None):
    """
    Runs inference. If filter_classes is list of ints (e.g. [1,4,7]), 
    only calculates acc on those classes (for SOC-3).
    """
    print(f"  -> Evaluating {split_name} {'(Filtered)' if filter_classes else ''}...")
    ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split=split_name, cache_memory=True)
    if len(ds) == 0: return 0.0, 0.0
    
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    criterion = PhysXLoss()
    criterion.ce_loss = nn.CrossEntropyLoss()
    
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            logits, recon, scatter, _ = model(img)
            
            # Loss is always calculated on batch (approx)
            loss = criterion(logits, recon, scatter, label, img)
            total_loss += loss.item()
            
            _, preds = torch.max(logits, 1)
            
            # Filter Logic for SOC-3
            if filter_classes is not None:
                # specific MSTAR classes: BMP2(1), BTR70(4), T72(7)
                mask = torch.isin(label, torch.tensor(filter_classes).to(DEVICE))
                if mask.sum() > 0:
                    correct += (preds[mask] == label[mask]).sum().item()
                    total += mask.sum().item()
            else:
                correct += (preds == label).sum().item()
                total += label.size(0)
            
    # Avoid div by zero if filter found nothing
    if total == 0: return 0.0, 0.0
    
    return total_loss / len(loader), 100 * correct / total

def save_table(df, filename):
    csv_path = f"{TABLES_DIR}/{filename}.csv"
    tex_path = f"{TABLES_DIR}/{filename}.tex"
    df.to_csv(csv_path, index=False)
    latex_code = df.to_latex(index=False, bold_rows=True, caption=filename.replace("_", " "))
    with open(tex_path, "w") as f:
        f.write(latex_code)
    print(f"  -> Saved {filename} (.csv & .tex)")

# --- TABLE GENERATORS ---

def generate_table_i_network_config(model):
    print("Generating Table I (Network Config)...")
    data = [
        ["CMKS_Branch1", "3x3", 1, "1x10x128x128"],
        ["CMKS_Branch2", "5x5", 1, "1x10x128x128"],
        ["CMKS_Branch3", "7x7", 1, "1x10x128x128"],
        ["Fusion", "Concat", "-", "30x128x128"]
    ]
    for i, cfg in enumerate(model.cfgs):
        k, _, c, _, s = cfg
        data.append([f"Ghost_Stage_{i+1}", f"Ghost({k}x{k})", s, f"{int(c*0.7)} channels"])
    data.append(["Physics_Branch", "Energy", "-", "1x128x128"])
    data.append(["Classifier", "Linear", "-", "10 Classes"])
    df = pd.DataFrame(data, columns=["Layer", "Kernel/Type", "Stride", "Output Dims"])
    save_table(df, "Table_I_Network_Config")

def generate_table_iii_soc10_config():
    print("Generating Table III (SOC-10 Dataset Config)...")
    # Data from Paper Table III
    data = [
        ["1-2S1", "B01", 299, 274],
        ["2-BMP2", "SN9563", 233, 195],
        ["3-BRDM2", "E-71", 298, 274],
        ["4-BTR-60", "K10yt7532", 256, 195],
        ["5-BTR-70", "C71", 233, 196],
        ["6-D7", "92v13015", 299, 274],
        ["7-T62", "A51", 299, 273],
        ["8-T72", "SN132", 232, 196],
        ["9-ZIL131", "E12", 299, 274],
        ["10-ZSU234", "D08", 299, 274],
        ["Total", "-", 2747, 2425]
    ]
    df = pd.DataFrame(data, columns=["Class", "Serial No.", "Training (17 deg)", "Testing (15 deg)"])
    save_table(df, "Table_III_SOC10_Config")

def generate_table_iv_soc3_config():
    print("Generating Table IV (SOC-3 Dataset Config)...")
    # Data from Paper Table IV
    data = [
        ["2-BMP2", "SN9563", 233, 195],
        ["5-BTR-70", "C71", 233, 196],
        ["8-T72", "SN132", 232, 196],
        ["Total", "-", 698, 587]
    ]
    df = pd.DataFrame(data, columns=["Class", "Serial No.", "Training (17 deg)", "Testing (15 deg)"])
    save_table(df, "Table_IV_SOC3_Config")

def generate_table_v_eoc_config():
    print("Generating Table V (EOC-VV-4 Dataset Config)...")
    # Data from Paper Table V
    data = [
        ["Train", "2-BMP2", "SN9563", 233],
        ["Train", "8-T72", "SN132", 232],
        ["Test", "2-BMP2", "SN9566", 196],
        ["Test", "2-BMP2", "C21", 196],
        ["Test", "8-T72", "SN812", 195],
        ["Test", "8-T72", "S7", 191]
    ]
    df = pd.DataFrame(data, columns=["Set", "Class", "Serial No.", "Count"])
    save_table(df, "Table_V_EOC_Config")

def generate_table_vi_soc10_results(model):
    print("Generating Table VI (SOC-10 Results)...")
    loss, acc = get_metrics(model, 'soc_test')
    data = [
        ["Traditional CNN", "Real", 0.3546, 96.25],
        ["VGG16", "Real", 0.0866, 97.65],
        ["ResNet18", "Real", 0.1469, 96.82],
        ["CVCNN", "Complex", 0.0655, 98.59],
        ["CV-Net", "Complex", 0.0225, 99.67],
        ["CRMC-Net (SOTA)", "Complex", 0.0114, 99.83],
        ["PhysX-MKS-Ghost (Ours)", "PhysX", f"{loss:.4f}", f"{acc:.2f}"]
    ]
    df = pd.DataFrame(data, columns=["Model", "Type", "Loss", "Accuracy (%)"])
    save_table(df, "Table_VI_SOC10_Results")

def generate_table_vii_soc3_results(model):
    print("Generating Table VII (SOC-3 Results)...")
    # SOC-3 uses only classes BMP2(1), BTR70(4), T72(7)
    # We filter the full SOC test set for these indices
    loss, acc = get_metrics(model, 'soc_test', filter_classes=[1, 4, 7])
    
    data = [
        ["Traditional CNN", "Real", 0.0273, 99.41],
        ["VGG16", "Real", 0.0315, 98.97],
        ["ResNet18", "Real", 0.0358, 99.05],
        ["CVCNN", "Complex", 0.0329, 99.26],
        ["CV-Net", "Complex", 0.0076, 99.83],
        ["CRMC-Net (SOTA)", "Complex", 0.0029, 100.00],
        ["PhysX-MKS-Ghost (Ours)", "PhysX", f"{loss:.4f}", f"{acc:.2f}"]
    ]
    df = pd.DataFrame(data, columns=["Model", "Type", "Loss", "Accuracy (%)"])
    save_table(df, "Table_VII_SOC3_Results")

def generate_table_viii_eoc_results(model):
    print("Generating Table VIII (EOC-VV-4 Results)...")
    loss, acc = get_metrics(model, 'eoc_2_test')
    data = [
        ["Traditional CNN", "Real", 0.3316, 91.33],
        ["VGG16", "Real", 0.3849, 88.68],
        ["ResNet18", "Real", 0.5477, 89.97],
        ["CVCNN", "Complex", 0.2451, 93.09],
        ["CV-Net", "Complex", 0.1727, 94.86],
        ["CRMC-Net (SOTA)", "Complex", 0.1383, 95.02],
        ["PhysX-MKS-Ghost (Ours)", "PhysX", f"{loss:.4f}", f"{acc:.2f}"]
    ]
    df = pd.DataFrame(data, columns=["Model", "Type", "Loss", "Accuracy (%)"])
    save_table(df, "Table_VIII_EOC_Results")

def generate_table_xi_efficiency(model):
    print("Generating Table XI (Efficiency)...")
    params_m, flops_g = count_flops_params(model)
    params_mb = params_m * 8.0 
    
    # Latency
    model.eval()
    dummy = torch.randn(1, 1, 128, 128, dtype=torch.complex64).to(DEVICE)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    start = time.time()
    for _ in range(200): _ = model(dummy)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    lat = ((time.time() - start) / 200) * 1000

    data = [
        ["ResNet18", 47.8, 1.27, "-"],
        ["VGG16", 138.4, 1.42, "-"],
        ["RMC-Net", 14.6, 0.22, "-"],
        ["CRMC-Net", 25.8, 1.31, "45.0"],
        ["PhysX-MKS-Ghost (Ours)", f"{params_mb:.2f}", f"{flops_g:.2f}", f"{lat:.2f}"]
    ]
    df = pd.DataFrame(data, columns=["Model", "Params (MB)", "FLOPs (G)", "Latency (ms)"])
    save_table(df, "Table_XI_Efficiency")

def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"Loading model from {CHECKPOINT}")
    model = PhysX_MKS_GhostNet(num_classes=10, width_mult=0.70, use_vlm=False, tiny=True).to(DEVICE)
    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    else:
        print("WARNING: No checkpoint found. Using random weights.")

    # Generate All Tables (1-11)
    generate_table_i_network_config(model)
    if os.path.exists("data/MSTAR_Combined"):
        generate_table_iii_soc10_config()  # NEW
        generate_table_iv_soc3_config()    # NEW
        generate_table_v_eoc_config()      # NEW
        generate_table_vi_soc10_results(model)
        generate_table_vii_soc3_results(model) # NEW
        generate_table_viii_eoc_results(model)
        
    generate_table_xi_efficiency(model)

    print("\n--- All Tables Generated in outputs/tables/ ---")

if __name__ == "__main__":
    main()