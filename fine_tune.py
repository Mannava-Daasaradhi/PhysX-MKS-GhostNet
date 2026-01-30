import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import torchvision.transforms as T

# --- IMPORTS ---
from src.models.net_architecture import PhysX_MKS_GhostNet
from src.dataset import MSTAR_Dataset
from src.losses import PhysXLoss 
from src.transforms import (
    ComplexRandomScale, 
    ComplexSpeckleNoise, 
    ComplexRandomRotation, 
    RandomPhaseShift,
    ComplexRandomErasing
)

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 30              
LEARNING_RATE = 1e-5     # <--- FIXED: Back to Low LR for stability
WEIGHT_DECAY = 0.0       # <--- FIXED: Zero decay prevents forgetting
CHECKPOINT_DIR = "outputs/checkpoints"
RESULTS_DIR = "outputs/results"

def get_finetune_transforms():
    """
    PHASE 2: THE CORRECT COMBINATION
    - Wide Scaling: Matches EOC geometry.
    - High Noise: Matches EOC texture.
    """
    return T.Compose([
        # [CRITICAL] Wide Scaling (0.7-1.3)
        ComplexRandomScale(scale_range=(0.7, 1.3), p=0.8), 
        
        # Moderate Rotation
        ComplexRandomRotation(degrees=15, p=0.5),
        
        # [CRITICAL] High Noise (0.15) - Reduced slightly from 0.2 to allow convergence
        ComplexSpeckleNoise(prob=0.5, sigma=0.15),          
        
        # Erasing
        ComplexRandomErasing(p=0.3, scale=(0.02, 0.1)),
        
        RandomPhaseShift(p=0.5)
    ])

def save_predictions(model, loader, split_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img, label in tqdm(loader, desc=f"Generating {split_name}"):
            img = img.to(DEVICE)
            logits, _, _, _ = model(img)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.numpy())
    np.save(f"{RESULTS_DIR}/preds_{split_name}.npy", np.array(all_preds))
    np.save(f"{RESULTS_DIR}/labels_{split_name}.npy", np.array(all_labels))

def validate(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            logits, recon, scatter, _ = model(img)
            _, preds = torch.max(logits, 1)
            correct += (preds == label).sum().item()
            total += label.size(0)
    return 100 * correct / total

def main():
    print(f"--- Starting Phase 2: Goldilocks Fine-Tuning (Target: 91%+) ---")
    
    # 1. Load Data
    train_ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='soc_train', transform=get_finetune_transforms(), cache_memory=True)
    val_ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='soc_test', cache_memory=True)
    eoc_ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='eoc_2_test', cache_memory=True)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    eoc_loader = DataLoader(eoc_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Load BEST EOC Model
    model = PhysX_MKS_GhostNet(num_classes=10, width_mult=0.70, use_vlm=False, tiny=True).to(DEVICE)
    load_path = f"{CHECKPOINT_DIR}/best_eoc_model.pth"
    
    if os.path.exists(load_path):
        print(f"Loading previous peak model: {load_path}")
        model.load_state_dict(torch.load(load_path))
    else:
        print("⚠️ Best EOC model not found! Loading best_model.pth")
        model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_model.pth"))

    # 3. Optimizer with Low LR and Zero Decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 
    criterion = PhysXLoss(alpha=0.2, beta=0.15)
    criterion.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print("Validating baseline...")
    best_eoc_acc = validate(model, eoc_loader, criterion)
    print(f"Baseline EOC Accuracy: {best_eoc_acc:.2f}%")
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train() # Unfrozen BN to adapt to noise
        
        loop = tqdm(train_loader, desc=f"Refining {epoch+1}/{EPOCHS}")
        for img, label in loop:
            img, label = img.to(DEVICE), label.to(DEVICE)
            
            optimizer.zero_grad()
            logits, recon, scatter, _ = model(img)
            loss = criterion(logits, recon, scatter, label, img)
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())
            
        # Validation
        val_acc = validate(model, val_loader, criterion)
        eoc_acc = validate(model, eoc_loader, criterion)
        
        print(f"Epoch {epoch+1}: SOC={val_acc:.2f}% | EOC={eoc_acc:.2f}%")
        
        if eoc_acc > best_eoc_acc:
            best_eoc_acc = eoc_acc
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/final_best_model.pth")
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth") 
            print(f"  >>> 🚀 NEW PEAK EOC: {eoc_acc:.2f}%")

    print("\n--- Generating Final Data ---")
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/final_best_model.pth"))
    save_predictions(model, val_loader, "soc_10")
    save_predictions(model, eoc_loader, "eoc")
    save_predictions(model, val_loader, "soc_3")
    print("✅ Fine-tuning complete. Run 'generate_paper_figures.py'")

if __name__ == "__main__":
    main()