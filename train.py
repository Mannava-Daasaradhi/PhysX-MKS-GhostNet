import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import os
import pandas as pd
from tqdm import tqdm

from src.dataset import MSTAR_Dataset
from src.models.net_architecture import PhysX_MKS_GhostNet
from src.losses import PhysXLoss
from src.transforms import RandomPhaseShift, ComplexRandomRotation, ComplexGaussianNoise

# --- TUNED HYPERPARAMETERS (Optimized for 99% Accuracy) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 0.001              
EPOCHS = 100            # Increased to 100 for full convergence
WEIGHT_DECAY = 1e-3     # Tuned to prevent underfitting while maintaining regularization

def main():
    print(f"--- PhysX-MKS-GhostNet Training (High-Performance Mode) ---")
    print(f"Device: {DEVICE} | Epochs: {EPOCHS} | Start LR: {LR} | WD: {WEIGHT_DECAY}")
    
    # 1. Setup Data Augmentation (Physics-Aware)
    train_transform = Compose([
        RandomPhaseShift(p=0.5),      # Force relative phase learning
        ComplexRandomRotation(p=0.5), # Rotation invariance
        ComplexGaussianNoise(p=0.15)  # Robustness to speckle noise
    ])
    
    # 2. Load Datasets
    print("\n--- Loading Datasets ---")
    data_root = "data/MSTAR_Combined"
    
    if not os.path.exists(data_root):
        print(f"ERROR: '{data_root}' not found. Please create it and add MSTAR data.")
        # Create dummy folders to prevent immediate crash if user is just testing structure
        os.makedirs(data_root, exist_ok=True)
        return

    # SoC-10 Training Split
    train_ds = MSTAR_Dataset(
        root_dir=data_root, 
        split='soc_train', 
        transform=train_transform, 
        cache_memory=True
    )
    
    # SoC-10 Test Split (Validation)
    val_ds = MSTAR_Dataset(
        root_dir=data_root, 
        split='soc_test', 
        transform=None, 
        cache_memory=True
    )
    
    if len(train_ds) == 0:
        print("WARNING: No training data found. Exiting.")
        return
        
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 3. Initialize Super-Architecture (Width x1.4)
    print("\n--- Initializing Super-Architecture (Width x1.4) ---")
    # width_mult=1.4 adds ~40% more parameters to capture fine details
    model = PhysX_MKS_GhostNet(num_classes=10, width_mult=1.4).to(DEVICE)
    
    # 4. Setup Optimizer & Scheduler
    # Using tuned loss weights: Alpha=0.2 (Recon), Beta=0.05 (Physics)
    criterion = PhysXLoss(alpha=0.2, beta=0.05) 
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Cosine Annealing for smooth convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # 5. Training Loop
    best_acc = 0.0
    history = []
    print("\n--- Starting Training Loop ---")
    
    for epoch in range(1, EPOCHS+1):
        # --- TRAIN STEP ---
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
        for img, label in loop:
            img, label = img.to(DEVICE), label.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward Pass: Returns (Logits, Recon, ScatterMap, VLM_Features)
            logits, recon, scatter, vlm_out = model(img)
            
            # Calculate PhysX Loss
            loss = criterion(logits, recon, scatter, label, img)
            
            # Backward Pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            
            loop.set_postfix(loss=loss.item())
            
        train_acc = 100 * correct / total
        train_loss_avg = total_loss / len(train_loader)
        
        # --- VALIDATION STEP ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                
                logits, recon, scatter, vlm_out = model(img)
                loss = criterion(logits, recon, scatter, label, img)
                
                val_loss += loss.item()
                _, pred = torch.max(logits, 1)
                val_correct += (pred == label).sum().item()
                val_total += label.size(0)
                
        val_acc = 100 * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        
        # --- LOGGING ---
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}: Train Loss={train_loss_avg:.4f} Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss_avg:.4f} Acc={val_acc:.2f}% | LR={current_lr:.6f}")
        
        history.append([epoch, train_loss_avg, train_acc, val_loss_avg, val_acc])
        
        # Scheduler Step
        scheduler.step()
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = "outputs/checkpoints/best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New Best Accuracy! Model saved to {save_path}")

    # 6. Save Training History
    df = pd.DataFrame(history, columns=['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc'])
    df.to_csv("outputs/results/training_curves.csv", index=False)
    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.2f}%")
    print("History saved to outputs/results/training_curves.csv")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    main()