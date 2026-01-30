import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import os
import cv2
from torch.utils.data import DataLoader

# --- IMPORTS FROM SRC ---
from src.dataset import MSTAR_Dataset, MSTAR_CLASSES
from src.models.net_architecture import PhysX_MKS_GhostNet
from src.models.complex_layers import ComplexConv2d

# --- CONFIGURATION ---
RESULTS_DIR = "outputs/results"
VIZ_DIR = "outputs/visualizations"
CHECKPOINT = "outputs/checkpoints/best_eoc_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(VIZ_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
colors = ["#2ecc71", "#e74c3c", "#3498db"]

# --- HELPER: HIGH-RES SALIENCY ---
def compute_saliency_map(model, input_tensor, class_idx=None):
    """
    Computes Input Gradients (Saliency) instead of GradCAM.
    This highlights exact scattering centers (pixels) rather than blobs.
    """
    model.eval()
    # Ensure gradients are tracked for the input
    input_tensor.requires_grad_()
    
    # Forward pass
    logits, _, _, _ = model(input_tensor)
    
    if class_idx is None:
        class_idx = torch.argmax(logits, dim=1)
    
    # Backward pass
    score = logits[0, class_idx]
    score.backward()
    
    # Get gradient magnitude
    # Input is Complex (B, 1, H, W) -> Grad is Complex
    grad = input_tensor.grad
    grad_mag = torch.abs(grad).squeeze().cpu().numpy()
    
    # Normalize (0 to 1)
    grad_mag = np.maximum(grad_mag, 0)
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
    
    return grad_mag

# --- PLOTTING ---
def safe_normalize_cm_columns(cm):
    col_sums = cm.sum(axis=0)[np.newaxis, :]
    col_sums[col_sums == 0] = 1
    return cm.astype('float') / col_sums

def plot_training_curves():
    csv_path = f"{RESULTS_DIR}/training_curves.csv"
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    l1 = ax1.plot(df['Epoch'], df['Val_Acc'], label='SOC-10', color=colors[2])
    l2 = ax1.plot(df['Epoch'], df['EoC_Acc'], label='EOC-2', color=colors[0], linestyle='--')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', fontweight='bold')
    l3 = ax2.plot(df['Epoch'], df['Val_Loss'], label='Loss', color=colors[1], alpha=0.4)
    ax1.legend(l1+l2+l3, [l.get_label() for l in l1+l2+l3], loc='center right')
    plt.title("Training Dynamics", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/Fig3_Training_Dynamics.png", dpi=300)
    plt.close(fig)

def plot_confusion_matrix(split_name, title_suffix):
    preds_path = f"{RESULTS_DIR}/preds_{split_name}.npy"
    labels_path = f"{RESULTS_DIR}/labels_{split_name}.npy"
    if not os.path.exists(preds_path): return
    preds, labels = np.load(preds_path), np.load(labels_path)
    acc = 100 * np.mean(preds == labels)
    cm = confusion_matrix(labels, preds)
    cm_norm = safe_normalize_cm_columns(cm)
    fig = plt.figure(figsize=(11, 9))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=MSTAR_CLASSES, yticklabels=MSTAR_CLASSES, vmin=0, vmax=1)
    plt.xlabel('Predicted Class (Sum=1)', fontweight='bold')
    plt.ylabel('True Class', fontweight='bold')
    plt.title(f"{title_suffix}\nAccuracy: {acc:.2f}% (Column Normalized)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/Fig4_CM_{split_name}.png", dpi=300)
    plt.close(fig)

def generate_complex_analysis_separate(model, val_ds, num_samples=10):
    indices = np.random.choice(len(val_ds), num_samples, replace=False)
    print(f"--- Generating {num_samples} Physics Maps ---")
    for i, idx in enumerate(indices):
        img_complex, label = val_ds[idx]
        class_name = MSTAR_CLASSES[label]
        mag = torch.abs(img_complex).squeeze().numpy()
        phase = torch.angle(img_complex).squeeze().numpy()
        real = img_complex.real.squeeze().numpy()
        imag = img_complex.imag.squeeze().numpy()
        components = [mag, phase, real, imag]
        titles = ["Magnitude", "Phase", "Real Part", "Imaginary Part"]
        cmaps = ['gray', 'twilight', 'seismic', 'seismic']
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, comp, title, cmap in zip(axes, components, titles, cmaps):
            ax.imshow(comp, cmap=cmap)
            ax.set_title(title, fontweight='bold')
            ax.axis('off')
        plt.suptitle(f"Sample {i+1}: {class_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{VIZ_DIR}/Fig5_Physics_Sample_{i+1}_{class_name}.png", dpi=300)
        plt.close(fig)

def generate_saliency_analysis_separate(model, val_ds, num_samples=10):
    """
    Generates Fig 6: High-Res Saliency Maps.
    """
    indices = np.random.choice(len(val_ds), num_samples, replace=False)
    print(f"--- Generating {num_samples} High-Res Saliency Maps ---")
    
    for i, idx in enumerate(indices):
        img_complex, label = val_ds[idx]
        class_name = MSTAR_CLASSES[label]
        input_tensor = img_complex.unsqueeze(0).to(DEVICE) # No requires_grad here, handled in helper
        
        # 1. Compute Saliency (Pixel Attention)
        saliency = compute_saliency_map(model, input_tensor)
        
        # --- HARD MASKING (Zero out bottom 50%) ---
        h_map, w_map = saliency.shape
        cutoff_row = int(h_map * 0.5)
        saliency[cutoff_row:, :] = 0
        # ------------------------------------------
        
        # Original Magnitude for display
        img_mag = torch.abs(img_complex).squeeze().numpy()
        img_mag_norm = (img_mag - img_mag.min()) / (img_mag.max() - img_mag.min())
        
        # Enhance Saliency Contrast for visibility (Gamma correction)
        saliency = np.power(saliency, 0.7) 
        
        # Colorize Saliency
        saliency_uint8 = np.uint8(255 * saliency)
        saliency_color = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_INFERNO)
        saliency_color = cv2.cvtColor(saliency_color, cv2.COLOR_BGR2RGB) / 255.0
        
        # Blend (Cleaner blend: 70% Map, 30% Original)
        superimposed = 0.3 * np.stack([img_mag_norm]*3, axis=-1) + 0.7 * saliency_color
        superimposed = np.clip(superimposed, 0, 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_mag_norm, cmap='gray')
        axes[0].set_title(f"Original ({class_name})", fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(superimposed)
        axes[1].set_title("Scattering Centers (Saliency)", fontweight='bold', color='darkred')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{VIZ_DIR}/Fig6_Saliency_Sample_{i+1}_{class_name}.png", dpi=300)
        plt.close(fig)
        print(f"  -> Saved Sample {i+1}")

def main():
    print("\n--- GENERATING FINAL PAPER FIGURES (SALIENCY) ---")
    model = PhysX_MKS_GhostNet(num_classes=10, width_mult=0.70, use_vlm=False, tiny=True).to(DEVICE)
    if os.path.exists(CHECKPOINT):
        state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)
        model.eval()
    else:
        print("❌ Error: Checkpoint not found.")
        return
    val_ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='soc_test', cache_memory=True)
    
    plot_training_curves()
    plot_confusion_matrix("soc_10", "SOC-10 (Standard)")
    plot_confusion_matrix("eoc", "EOC-2 (Robustness)")
    generate_complex_analysis_separate(model, val_ds, num_samples=10) 
    generate_saliency_analysis_separate(model, val_ds, num_samples=10)  # Using Saliency now
    print(f"\n✅ All figures ready: {os.path.abspath(VIZ_DIR)}")

if __name__ == "__main__":
    main()