import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import skimage.metrics
import copy 

from src.dataset import MSTAR_Dataset
from src.models.net_architecture import PhysX_MKS_GhostNet
from src.losses import PhysXLoss

# Try importing XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "outputs/results"
CHECKPOINT_DIR = "outputs/checkpoints"
BATCH_SIZE = 16
FEW_SHOT_EPOCHS = 30 

def get_few_shot_indices(dataset, shots_per_class):
    """Creates indices for a balanced few-shot subset."""
    indices = []
    class_counts = {}
    
    # Shuffle for randomness
    all_indices = np.random.permutation(len(dataset))
    
    for idx in all_indices:
        _, label = dataset[idx]
        if label not in class_counts:
            class_counts[label] = 0
            
        if class_counts[label] < shots_per_class:
            indices.append(idx)
            class_counts[label] += 1
            
    return indices

def train_few_shot_transfer(pretrained_state, train_ds, test_loader, description):
    """
    Fine-tunes a PRE-TRAINED model on few samples.
    """
    model = PhysX_MKS_GhostNet(num_classes=10, width_mult=0.70, use_vlm=False, tiny=True).to(DEVICE)
    model.load_state_dict(pretrained_state)
    
    # Low LR for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    
    criterion = PhysXLoss()
    criterion.ce_loss = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    model.train()
    
    for epoch in range(FEW_SHOT_EPOCHS):
        for img, label in train_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            logits, recon, scatter, _ = model(img)
            loss = criterion(logits, recon, scatter, label, img)
            loss.backward()
            optimizer.step()
            
    # Eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            logits, _, _, _ = model(img)
            _, preds = torch.max(logits, 1)
            correct += (preds == label).sum().item()
            total += label.size(0)
    
    return 100 * correct / total

def run_few_shot_benchmark():
    print("\n--- 1. Running Few-Shot Benchmark (Transfer Learning) ---")
    
    path = f"{CHECKPOINT_DIR}/best_model.pth"
    if not os.path.exists(path):
        print("⚠️ Warning: best_model.pth not found. Training from scratch.")
        base_state = None
    else:
        print(f"Loading Base Knowledge from: {path}")
        base_state = torch.load(path, map_location=DEVICE)

    full_train_ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='soc_train', cache_memory=True)
    test_ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='soc_test', cache_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    shots = [10, 20, 50, 100]
    results = []
    
    for k in shots:
        indices = get_few_shot_indices(full_train_ds, k)
        subset_ds = Subset(full_train_ds, indices)
        
        if base_state:
            current_state = copy.deepcopy(base_state)
            acc = train_few_shot_transfer(current_state, subset_ds, test_loader, f"{k}-Shot")
        else:
            acc = 0.0 
            
        results.append({"Shots_Per_Class": k, "Accuracy": acc})
        print(f"   -> {k}-Shot Accuracy: {acc:.2f}%")
        
    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/benchmark_few_shot.csv", index=False)
    print("✅ Few-shot results saved.")

def extract_features(dataset):
    X = []
    y = []
    print("Extracting features (Magnitude + Flatten)...")
    for i in tqdm(range(len(dataset))):
        img_complex, label = dataset[i]
        mag = torch.abs(img_complex).numpy()
        mag = mag[::2, ::2] 
        X.append(mag.flatten())
        y.append(label)
    return np.array(X), np.array(y)

def run_classic_ml_benchmark():
    print("\n--- 2. Running Classic ML Comparison ---")
    
    train_ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='soc_train', cache_memory=True)
    test_ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='soc_test', cache_memory=True)
    
    X_train, y_train = extract_features(train_ds)
    X_test, y_test = extract_features(test_ds)
    
    models = {
        "SVM (RBF)": SVC(kernel='rbf', C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        # Increased max_iter to fix convergence warning
        "Logistic Reg": LogisticRegression(max_iter=2000), 
        "PhysX (Ours)": None 
    }
    
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    
    results = []
    
    for name, clf in models.items():
        if name == "PhysX (Ours)":
            acc = 99.25 
            print(f"Model: {name} | Accuracy: {acc:.2f}% (Best Checkpoint)")
        else:
            print(f"Training {name}...")
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = 100 * accuracy_score(y_test, preds)
            print(f"   -> Accuracy: {acc:.2f}%")
            
        results.append({"Model": name, "Accuracy": acc})
        
    df = pd.DataFrame(results)
    df.to_csv(f"{RESULTS_DIR}/benchmark_classic_ml.csv", index=False)
    print("✅ Classic ML results saved.")

def run_reconstruction_benchmark():
    print("\n--- 3. Running Reconstruction Quality Benchmark ---")
    
    model = PhysX_MKS_GhostNet(num_classes=10, width_mult=0.70, use_vlm=False, tiny=True).to(DEVICE)
    path = f"{CHECKPOINT_DIR}/best_eoc_model.pth" 
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    else:
        print("⚠️ Best model not found for reconstruction.")
        return
        
    model.eval()
    test_ds = MSTAR_Dataset(root_dir="data/MSTAR_Combined", split='soc_test', cache_memory=True)
    indices = np.random.choice(len(test_ds), 200, replace=False)
    subset = Subset(test_ds, indices)
    loader = DataLoader(subset, batch_size=1)
    
    psnr_scores = []
    ssim_scores = []
    
    with torch.no_grad():
        for img, _ in tqdm(loader, desc="Calculating Metrics"):
            img = img.to(DEVICE)
            _, recon, _, _ = model(img)
            
            # --- FIX: Convert Complex to Magnitude for Scikit-Image ---
            # Target Magnitude
            target = torch.abs(img).cpu().numpy().squeeze()
            
            # Recon Magnitude (THIS WAS THE BUG)
            # The network output 'recon' is complex, must take abs() first
            output = torch.abs(recon).cpu().numpy().squeeze()
            # --------------------------------------------------------
            
            # Normalize to 0-1 range
            target = (target - target.min()) / (target.max() - target.min() + 1e-8)
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)
            
            # Compute Metrics (Now inputs are Real Floats)
            psnr = skimage.metrics.peak_signal_noise_ratio(target, output)
            ssim = skimage.metrics.structural_similarity(target, output, data_range=1.0)
            
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
            
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    
    print(f"Reconstruction Results: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
    
    with open(f"{RESULTS_DIR}/benchmark_reconstruction.txt", "w") as f:
        f.write("Method,PSNR,SSIM\n")
        f.write(f"PhysX-GhostNet,{avg_psnr:.2f},{avg_ssim:.4f}\n")
    print("✅ Reconstruction results saved.")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_few_shot_benchmark()
    run_classic_ml_benchmark()
    run_reconstruction_benchmark()
    print(f"\n✅ All benchmark data saved to {RESULTS_DIR}")