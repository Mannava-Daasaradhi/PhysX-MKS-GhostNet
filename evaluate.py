import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.dataset import MSTAR_Dataset
from src.models.net_architecture import PhysX_MKS_GhostNet

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
CHECKPOINT_PATH = "outputs/checkpoints/best_model.pth"
RESULTS_DIR = "outputs/results"

def evaluate_split(model, split_name, dataset_name="MSTAR_Combined"):
    """
    Runs inference on a specific data split and returns metrics.
    """
    print(f"\n--- Evaluating Split: {split_name} ---")
    
    # 1. Load Data
    ds = MSTAR_Dataset(
        root_dir=f"data/{dataset_name}", 
        split=split_name, 
        transform=None, 
        cache_memory=True
    )
    
    if len(ds) == 0:
        print(f"WARNING: No data found for {split_name}. Skipping.")
        return None, None, None

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. Inference
    model.eval()
    all_preds = []
    all_labels = []
    all_features = [] # For t-SNE later
    
    with torch.no_grad():
        for img, label in tqdm(loader, desc=f"Testing {split_name}"):
            img = img.to(DEVICE)
            
            # Forward pass
            logits, _, _, _ = model(img)
            
            # Get predictions
            _, pred = torch.max(logits, 1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.numpy())
            # We assume logits can serve as features for t-SNE, or we could extract earlier
            all_features.extend(logits.cpu().numpy())

    # 3. Compute Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"  >>> Accuracy: {acc*100:.2f}%")
    
    # 4. Save Raw Results for Figure Generation
    np.save(f"{RESULTS_DIR}/preds_{split_name}.npy", np.array(all_preds))
    np.save(f"{RESULTS_DIR}/labels_{split_name}.npy", np.array(all_labels))
    np.save(f"{RESULTS_DIR}/features_{split_name}.npy", np.array(all_features))
    
    return acc, all_labels, all_preds

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load Model
    print(f"Loading Model from {CHECKPOINT_PATH}...")
    if not os.path.exists(CHECKPOINT_PATH):
        print("ERROR: Checkpoint not found. Run train.py first!")
        return

    model = PhysX_MKS_GhostNet(num_classes=10, width_mult=1.4).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    
    # 2. Run The "Gauntlet" (Stress Tests)
    results = []
    
    # Test A: Standard Operating Conditions (SoC)
    acc_soc, _, _ = evaluate_split(model, "soc_test")
    if acc_soc is not None:
        results.append(["SoC-10", "Standard (15 deg)", acc_soc * 100])

    # Test B: Extended Operating Conditions 1 (Depression Angle)
    acc_eoc1, _, _ = evaluate_split(model, "eoc_1_test")
    if acc_eoc1 is not None:
        results.append(["EoC-1", "Depression Angle (30/45 deg)", acc_eoc1 * 100])

    # Test C: Extended Operating Conditions 2 (Version Variants)
    acc_eoc2, _, _ = evaluate_split(model, "eoc_2_test")
    if acc_eoc2 is not None:
        results.append(["EoC-2", "Version Variants (BMP2-C21, T72-A32...)", acc_eoc2 * 100])
        
    # 3. Generate Summary Tables (CSV)
    df = pd.DataFrame(results, columns=["Test_Set", "Condition", "Accuracy (%)"])
    print("\n--- Final Evaluation Report ---")
    print(df)
    
    # Save Table III / IV Data
    df.to_csv(f"{RESULTS_DIR}/evaluation_summary.csv", index=False)
    print(f"\nSummary saved to {RESULTS_DIR}/evaluation_summary.csv")

if __name__ == "__main__":
    main()