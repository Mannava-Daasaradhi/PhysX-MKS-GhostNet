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
from src.losses import PhysXLoss 

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 100             
LEARNING_RATE = 1e-3     
WEIGHT_DECAY = 1e-4      
LABEL_SMOOTHING = 0.1    

# --- NEW CONFIGS FOR FIXES ---
PATIENCE = 15            # Stop if no improvement for 15 epochs (Issue 6)
RESUME_PATH = None       # Set to "outputs/checkpoints/last_model.pth" to resume (Issue 4)

# --- SAVE PATHS ---
CHECKPOINT_DIR = "outputs/checkpoints"
RESULTS_DIR = "outputs/results"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- HELPER CLASSES ---

class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def save_checkpoint(state, filename="last_model.pth"):
    """Saves model, optimizer, and scheduler state."""
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, path)

def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    for img, label in loop:
        img, label = img.to(DEVICE), label.to(DEVICE)
        
        optimizer.zero_grad()
        
        logits, recon, scatter, _ = model(img)
        
        loss = criterion(logits, recon, scatter, label, img)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
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
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Model Initialization
    print("Initializing PhysX-MKS-GhostNet (Target: <0.3M Params)...")
    model = PhysX_MKS_GhostNet(num_classes=10, width_mult=0.70, use_vlm=False, tiny=True).to(DEVICE)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {params/1e6:.4f} M")

    # 3. Setup
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    criterion = PhysXLoss(alpha=0.2, beta=0.15)
    criterion.ce_loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    # --- ISSUE 6 FIX: Early Stopping Init ---
    early_stopper = EarlyStopping(patience=PATIENCE)

    # --- ISSUE 4 FIX: Resume Logic ---
    start_epoch = 0
    best_acc = 0.0
    history = {'Epoch': [], 'Train_Loss': [], 'Train_Acc': [], 'Val_Loss': [], 'Val_Acc': []}

    if RESUME_PATH and os.path.isfile(RESUME_PATH):
        print(f"Loading checkpoint from {RESUME_PATH}...")
        checkpoint = torch.load(RESUME_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resuming from Epoch {start_epoch+1}")

    # 4. Training Loop
    for epoch in range(start_epoch, EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, epoch)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}%")
        
        # Update History
        history['Epoch'].append(epoch+1)
        history['Train_Loss'].append(train_loss)
        history['Train_Acc'].append(train_acc)
        history['Val_Loss'].append(val_loss)
        history['Val_Acc'].append(val_acc)
        
        # --- ISSUE 4 FIX: Save Last Checkpoint (Every Epoch) ---
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, filename="last_model.pth")

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, filename="best_model.pth")
            print(f"  >>> New Best Model Saved! ({val_acc:.2f}%)")
        
        pd.DataFrame(history).to_csv(f"{RESULTS_DIR}/training_curves.csv", index=False)

        # --- ISSUE 6 FIX: Early Stopping Check ---
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print("\n--- Training Complete ---")
    print(f"Top Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
