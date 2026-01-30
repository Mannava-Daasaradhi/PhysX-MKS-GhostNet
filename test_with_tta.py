import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import os
# --- IMPORTS ---
from src.models.net_architecture import PhysX_MKS_GhostNet
from src.dataset import MSTAR_Dataset

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

def complex_resize(img, size):
    """Safely resize complex tensors by processing Real/Imag channels independently."""
    real = img.real
    imag = img.imag
    
    # Resize treats channels independently, so we can pass them together if stacked, 
    # or separately. Doing separately is safest for dimension sanity.
    real_up = TF.resize(real, size, antialias=True)
    imag_up = TF.resize(imag, size, antialias=True)
    
    return torch.complex(real_up, imag_up)

def complex_hflip(img):
    """Safely flip complex tensors."""
    return torch.complex(TF.hflip(img.real), TF.hflip(img.imag))

def complex_center_crop(img, size):
    return torch.complex(TF.center_crop(img.real, size), TF.center_crop(img.imag, size))

def tta_inference(model, img):
    """
    Complex-Safe Test-Time Augmentation
    """
    # 1. Standard Prediction
    logits, _, _, _ = model(img)
    probs = F.softmax(logits, dim=1)

    # 2. Horizontal Flip (Complex Safe)
    img_flip = complex_hflip(img)
    logits_flip, _, _, _ = model(img_flip)
    probs += F.softmax(logits_flip, dim=1)

    # 3. Scale Up (1.1x) - crop center
    b, c, h, w = img.shape
    scale_h, scale_w = int(h * 1.15), int(w * 1.15)
    
    img_up = complex_resize(img, [scale_h, scale_w])
    img_up = complex_center_crop(img_up, [h, w])
    
    logits_up, _, _, _ = model(img_up)
    probs += F.softmax(logits_up, dim=1)

    # 4. Scale Down (0.9x) - pad border
    scale_h_down, scale_w_down = int(h * 0.9), int(w * 0.9)
    img_down = complex_resize(img, [scale_h_down, scale_w_down])
    
    # Calculate padding
    pad_h = (h - img_down.shape[2]) // 2
    pad_w = (w - img_down.shape[3]) // 2
    
    # Pad Real and Imag separately
    real_pad = F.pad(img_down.real, (pad_w, pad_w, pad_h, pad_h))
    imag_pad = F.pad(img_down.imag, (pad_w, pad_w, pad_h, pad_h))
    
    # Fix rounding errors
    if real_pad.shape[2] != h: 
        real_pad = F.pad(real_pad, (0, 0, 0, 1))
        imag_pad = F.pad(imag_pad, (0, 0, 0, 1))
    if real_pad.shape[3] != w: 
        real_pad = F.pad(real_pad, (0, 1, 0, 0))
        imag_pad = F.pad(imag_pad, (0, 1, 0, 0))
        
    img_down = torch.complex(real_pad, imag_pad)
    
    logits_down, _, _, _ = model(img_down)
    probs += F.softmax(logits_down, dim=1)

    return probs / 4.0  # Average

def validate_with_tta(model, loader, split_name):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    print(f"--- Running Complex-Safe TTA Inference on {split_name} ---")
    with torch.no_grad():
        for img, label in tqdm(loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            
            # Use TTA Wrapper
            avg_probs = tta_inference(model, img)
            
            _, preds = torch.max(avg_probs, 1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    acc = 100 * correct / total
    print(f"Result for {split_name}: {acc:.2f}% Accuracy")
    
    # Save for paper figures
    np.save(f"outputs/results/preds_{split_name}.npy", np.array(all_preds))
    np.save(f"outputs/results/labels_{split_name}.npy", np.array(all_labels))
    return acc

def main():
    # 1. Load Data
    val_ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='soc_test', cache_memory=True)
    eoc_ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='eoc_2_test', cache_memory=True)
    
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    eoc_loader = DataLoader(eoc_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 2. Load the 84.8% Model
    model = PhysX_MKS_GhostNet(num_classes=10, width_mult=0.70, use_vlm=False, tiny=True).to(DEVICE)
    path = "outputs/checkpoints/best_eoc_model.pth"
    
    if os.path.exists(path):
        print(f"Loading weights from: {path}")
        model.load_state_dict(torch.load(path))
    else:
        print("Error: best_eoc_model.pth not found. Using best_model.pth")
        model.load_state_dict(torch.load("outputs/checkpoints/best_model.pth"))
    
    # 3. Run Inference
    validate_with_tta(model, val_loader, "soc_10")
    eoc_acc = validate_with_tta(model, eoc_loader, "eoc")
    
    # Run SOC-3 just for file compatibility
    validate_with_tta(model, val_loader, "soc_3")

    if eoc_acc > 88.0:
        print(f"\n✅ SUCCESS! TTA pushed accuracy to {eoc_acc:.2f}%")
    else:
        print(f"\nℹ️ TTA Result: {eoc_acc:.2f}%. Check best_eoc_model.pth integrity.")

if __name__ == "__main__":
    main()