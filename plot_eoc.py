import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# --- CONFIGURATION ---
RESULTS_DIR = "outputs/results"
VIZ_DIR = "outputs/visualizations"
EOC_LABELS_PATH = f"{RESULTS_DIR}/labels_eoc.npy"
EOC_PREDS_PATH = f"{RESULTS_DIR}/preds_eoc.npy"

# Standard MSTAR Classes (Index 0-9)
MSTAR_CLASSES = ['2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU234']

# Setup Plotting Style
os.makedirs(VIZ_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4) # Slightly larger font for compact plot

def safe_normalize_cm_columns(cm):
    """Normalizes the confusion matrix by column (Prediction Precision)."""
    col_sums = cm.sum(axis=0)[np.newaxis, :]
    col_sums[col_sums == 0] = 1
    return cm.astype('float') / col_sums

def plot_compact_eoc():
    print("--- Generating Compact EOC Confusion Matrix ---")
    
    # 1. Load Data
    if not os.path.exists(EOC_LABELS_PATH) or not os.path.exists(EOC_PREDS_PATH):
        print(f"❌ Error: Files not found in {RESULTS_DIR}")
        print("   Please run 'generate_target_results.py' first to create the data.")
        return

    labels = np.load(EOC_LABELS_PATH)
    preds = np.load(EOC_PREDS_PATH)
    
    # 2. Identify Active Classes (Classes that actually exist in the Ground Truth)
    # We find indices where count > 0 in the labels
    unique_labels = np.unique(labels)
    active_indices = sorted(unique_labels)
    
    # Filter Class Names
    active_class_names = [MSTAR_CLASSES[i] for i in active_indices]
    
    print(f"Found {len(active_indices)} active classes in EOC: {active_class_names}")
    
    # 3. Compute Confusion Matrix
    # We compute it for ALL classes first to ensure indices align, then slice it
    cm_full = confusion_matrix(labels, preds, labels=range(len(MSTAR_CLASSES)))
    
    # Slice the matrix to keep only rows/cols of active indices
    # np.ix_ allows us to slice both dimensions at once
    cm_compact = cm_full[np.ix_(active_indices, active_indices)]
    
    # 4. Normalize (Column Normalization)
    cm_norm = safe_normalize_cm_columns(cm_compact)
    
    # Calculate subset accuracy
    # (Only for the samples in this view)
    acc = 100 * np.mean(labels == preds)

    # 5. Plot
    plt.figure(figsize=(8, 7)) # Smaller figure size since fewer classes
    
    # Create Heatmap
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=active_class_names, 
                yticklabels=active_class_names,
                vmin=0, vmax=1, cbar=True)
    
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.ylabel('True Class', fontweight='bold')
    plt.title(f"EOC-2 Robustness (Active Classes Only)\nAccuracy: {acc:.2f}%", fontweight='bold')
    
    # Save
    save_path = f"{VIZ_DIR}/Fig4_CM_eoc_compact.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved Compact Matrix: {save_path}")

if __name__ == "__main__":
    plot_compact_eoc()