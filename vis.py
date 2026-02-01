import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import cv2
from torch.utils.data import DataLoader

# --- IMPORTS ---
from src.dataset import MSTAR_Dataset, MSTAR_CLASSES
from src.models.net_architecture import PhysX_MKS_GhostNet

# --- CONFIGURATION ---
CHECKPOINT = "outputs/checkpoints/best_eoc_model.pth"
SAVE_DIR = "outputs/visualizations/reconstructions"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

def normalize_image(img):
    """Normalizes image data to 0-1 range for plotting."""
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img

def visualize_all_classes():
    print("--- Generating Reconstruction Visualizations for All Classes ---")
    
    # 1. Load Model
    print(f"Loading weights from: {CHECKPOINT}")
    model = PhysX_MKS_GhostNet(num_classes=10, width_mult=0.70, use_vlm=False, tiny=True).to(DEVICE)
    
    if os.path.exists(CHECKPOINT):
        # weights_only=False required for complex support
        state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)
    else:
        print("❌ Error: Checkpoint not found! Run training first.")
        return

    model.eval()
    
    # 2. Load Dataset
    test_ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='soc_test', cache_memory=True)
    
    # 3. Find one sample for each class
    class_samples = {}
    found_count = 0
    
    print("Searching for representative samples...")
    for i in range(len(test_ds)):
        img, label = test_ds[i]
        class_name = MSTAR_CLASSES[label]
        
        if class_name not in class_samples:
            class_samples[class_name] = img
            found_count += 1
            
        if found_count == 10:
            break
            
    # 4. Generate Visualizations
    fig, axes = plt.subplots(10, 2, figsize=(8, 20))
    
    print("Processing reconstructions...")
    
    sorted_classes = sorted(list(class_samples.keys()))
    
    with torch.no_grad():
        for i, class_name in enumerate(sorted_classes):
            img = class_samples[class_name]
            
            # Forward Pass
            input_tensor = img.unsqueeze(0).to(DEVICE)
            _, recon, _, _ = model(input_tensor)
            
            # --- PROCESS ORIGINAL ---
            orig_mag = torch.abs(img).squeeze().numpy()
            orig_disp = normalize_image(orig_mag)
            
            # --- PROCESS RECONSTRUCTION ---
            recon_mag = torch.abs(recon).squeeze().cpu().numpy()
            recon_disp = normalize_image(recon_mag)
            
            # --- PLOT ---
            ax_orig = axes[i, 0]
            ax_orig.imshow(orig_disp, cmap='gray')
            ax_orig.set_ylabel(class_name, fontsize=12, fontweight='bold', rotation=90)
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            if i == 0: ax_orig.set_title("Original Input\n(Magnitude)", fontweight='bold')
            
            ax_recon = axes[i, 1]
            ax_recon.imshow(recon_disp, cmap='inferno')
            ax_recon.set_xticks([])
            ax_recon.set_yticks([])
            if i == 0: ax_recon.set_title("Physics Reconstruction\n(Feature Energy)", fontweight='bold')

    plt.tight_layout()
    save_path = f"{SAVE_DIR}/All_Classes_Reconstruction.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved summary grid: {save_path}")
    
    # 5. Save Individual Pairs (Fixed Loop)
    print("Saving individual class pairs...")
    
    # [FIX] Added torch.no_grad() block here to prevent gradient error
    with torch.no_grad():
        for class_name in class_samples:
            img = class_samples[class_name]
            input_tensor = img.unsqueeze(0).to(DEVICE)
            _, recon, _, _ = model(input_tensor)
            
            # [FIX] Added .detach() just in case, though no_grad handles it
            orig_mag = normalize_image(torch.abs(img).squeeze().numpy())
            recon_mag = normalize_image(torch.abs(recon).detach().squeeze().cpu().numpy())
            
            fig_single, ax_single = plt.subplots(1, 2, figsize=(8, 4))
            ax_single[0].imshow(orig_mag, cmap='gray')
            ax_single[0].set_title(f"Original ({class_name})")
            ax_single[0].axis('off')
            
            ax_single[1].imshow(recon_mag, cmap='inferno')
            ax_single[1].set_title("Reconstruction")
            ax_single[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}/Recon_{class_name}.png", dpi=150)
            plt.close(fig_single)
            
    print("✅ All individual pairs saved.")

if __name__ == "__main__":
    visualize_all_classes()