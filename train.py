import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import your model and dataset
from src.models.net_architecture import PhysX_MKS_GhostNet
from src.dataset import MSTAR_Dataset
# FIX 1: Correct Import Name
from src.losses import PhysXLoss 

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16          # Reduced batch size slightly for stability with complex grads
EPOCHS = 50              # 50 Epochs is usually enough for MSTAR
LEARNING_RATE = 1e-3     # Standard Adam LR
WEIGHT_DECAY = 1e-4      # Regularization

# --- SAVE PATHS ---
CHECKPOINT_DIR = "outputs/checkpoints"
RESULTS_DIR = "outputs/results"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    for img, label in loop:
        img, label = img.to(DEVICE), label.to(DEVICE)
        
        # Zero Gradients
        optimizer.zero_grad()
        
        # Forward Pass
        # Returns: logits, reconstructed_img, scattering_map, vlm_features
        logits, recon, scatter, _ = model(img)
        
        # Calculate Loss
        # FIX 2: Updated arguments to match src/losses.py signature:
        # forward(logits, recon, scatter, targets, inputs)
        loss = criterion(logits, recon, scatter, label, img)
        
        # Backward Pass (Handle Complex Gradients automatically via PyTorch)
        loss.backward()
        
        # Gradient Clipping (Crucial for Complex Networks to prevent exploding grads)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, preds = torch.max(logits, 1)
        correct += (preds == label).sum().item()
        total += label.size(0)
        
        loop.set_postfix(loss=loss.item(), acc=correct/total)
        
    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img, label in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            img, label = img.to(DEVICE), label.to(DEVICE)
            
            logits, recon, scatter, _ = model(img)
            
            # FIX 3: Updated arguments here as well
            loss = criterion(logits, recon, scatter, label, img)
            
            running_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            
    acc = 100 * correct / total
    return running_loss / len(loader), acc

def main():
    print(f"--- Starting Training on {DEVICE} ---")
    
    # 1. Dataset
    print("Loading Datasets...")
    train_ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='soc_train', cache_memory=True)
    val_ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='soc_test', cache_memory=True)
    
    if len(train_ds) == 0:
        print("Error: No training data found in data/MSTAR_Combined.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Model Initialization (LIGHTWEIGHT MODE)
    print("Initializing PhysX-MKS-GhostNet...")
    # width_mult=1.0 -> Standard GhostNet width (~0.3M params)
    # use_vlm=False -> Disable LLM Projector (saves ~0.2M params)
    model = PhysX_MKS_GhostNet(num_classes=10, width_mult=1.0, use_vlm=False).to(DEVICE)
    
    # Count params to verify
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {params/1e6:.2f} M (Target: ~0.3-0.4 M)")

    # 3. Setup
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # FIX 4: Use PhysXLoss with your tuned weights
    criterion = PhysXLoss(alpha=0.2, beta=0.05)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    # 4. Training Loop
    best_acc = 0.0
    history = {'Epoch': [], 'Train_Loss': [], 'Train_Acc': [], 'Val_Loss': [], 'Val_Acc': []}
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, epoch)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}%")
        
        # Log History
        history['Epoch'].append(epoch+1)
        history['Train_Loss'].append(train_loss)
        history['Train_Acc'].append(train_acc)
        history['Val_Loss'].append(val_loss)
        history['Val_Acc'].append(val_acc)
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
            print(f"  >>> New Best Model Saved! ({val_acc:.2f}%)")
            
        # Save History CSV (Live updates)
        pd.DataFrame(history).to_csv(f"{RESULTS_DIR}/training_curves.csv", index=False)

    print("\n--- Training Complete ---")
    print(f"Top Accuracy: {best_acc:.2f}%")
    print(f"Saved to {CHECKPOINT_DIR}/best_model.pth")

if __name__ == "__main__":
    main()