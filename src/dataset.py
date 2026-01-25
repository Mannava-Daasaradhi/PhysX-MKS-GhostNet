import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import parse_phoenix_header, read_phoenix_data

# Standard MSTAR Class Mapping
MSTAR_CLASSES = ['2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU234']

class MSTAR_Dataset(Dataset):
    """
    PhysX-MKS-GhostNet Data Engine.
    Handles Complex I/O and strict SoC/EoC splitting protocols.
    """
    def __init__(self, root_dir, split='soc_train', transform=None, cache_memory=True, subset_fraction=1.0):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.cache_memory = cache_memory
        self.subset_fraction = subset_fraction
        self.classes = MSTAR_CLASSES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # 1. Gather File Paths based on Split Logic
        print(f"--- Scanning Dataset for {split} (Usage: {subset_fraction*100}%) ---")
        self.samples = self._load_sample_paths()
        
        # 2. Apply Few-Shot Subset (Deterministically)
        # Critical for Table VI: Few-Shot Performance
        if self.subset_fraction < 1.0 and len(self.samples) > 0:
            random.seed(42) # Fixed seed for reproducibility
            random.shuffle(self.samples)
            cutoff = int(len(self.samples) * self.subset_fraction)
            self.samples = self.samples[:cutoff]
            print(f"  [Subset] Reduced to {len(self.samples)} samples.")

        # 3. Cache Data to RAM (High-Performance Mode)
        self.data_cache = []
        if self.cache_memory and len(self.samples) > 0:
            print(f"  [Cache] Loading {len(self.samples)} complex images to RAM...")
            self.data_cache = self._cache_images()
            print(f"  [Cache] Complete. Valid samples: {len(self.data_cache)}")

    def _load_sample_paths(self):
        samples = []
        class_counts = {c: 0 for c in self.classes}
        
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                f_path = os.path.join(root, file)
                
                # Filter junk files
                if file.startswith('.'): continue
                if file.lower().endswith(('.jpg', '.png', '.txt', '.py', '.md', '.csv')): continue
                
                path_upper = f_path.upper()
                path_parts = path_upper.replace('\\', '/').split('/')
                
                # Identify Class from file path
                found_class = None
                for cls in self.classes:
                    if cls in path_parts:
                        found_class = cls
                        break
                
                # Robust Fallback Mappings for inconsistent folder naming
                if found_class is None:
                    if 'BTR_70' in path_parts: found_class = 'BTR70'
                    elif 'BRDM_2' in path_parts: found_class = 'BRDM2'
                    elif 'BTR_60' in path_parts: found_class = 'BTR60'
                    elif 'ZSU_23_4' in path_parts: found_class = 'ZSU234'
                    elif '2S1' in path_parts: found_class = '2S1'
                
                if found_class is None: continue

                # --- SPLIT LOGIC (Strict adherence to Table II) ---
                is_valid_split = False
                
                # 1. SoC (Standard Operating Conditions)
                if self.split == 'soc_train':
                    # Training: 17 Degree depression angle
                    if '17_DEG' in path_upper or ('TRAIN' in path_upper and '15_DEG' not in path_upper):
                        is_valid_split = True
                elif self.split == 'soc_test':
                    # Testing: 15 Degree depression angle
                    if '15_DEG' in path_upper or ('TEST' in path_upper and '17_DEG' not in path_upper):
                        is_valid_split = True
                
                # 2. EoC-1 (Large Depression Angle Changes)
                # Training is done on SoC-17, Testing on 30 and 45
                elif self.split == 'eoc_1_test':
                    if '30_DEG' in path_upper or '45_DEG' in path_upper:
                        is_valid_split = True
                
                # 3. EoC-2 (Version Variants)
                # Testing specifically on variants not in standard training (e.g., BMP2-C21)
                elif self.split == 'eoc_2_test':
                    # BMP2-C21 / 9566 are the primary stress tests
                    if found_class == 'BMP2' and ('C21' in path_upper or '9566' in path_upper):
                        is_valid_split = True
                    # T72 Variants (S7, A32, A64)
                    elif found_class == 'T72' and ('S7' in path_parts or 'A32' in path_parts or 'A64' in path_parts):
                        is_valid_split = True
                    # If specific variants aren't explicitly named, exclude 17/15 deg to find "other" test files
                    elif '17_DEG' not in path_upper and '15_DEG' not in path_upper:
                        is_valid_split = True

                if is_valid_split:
                    samples.append((f_path, self.class_to_idx[found_class]))
                    class_counts[found_class] += 1
        
        print(f"  [Debug] Found {len(samples)} files.")
        print(f"  [Debug] Distribution: {class_counts}")
        return samples

    def _cache_images(self):
        cache = []
        for path, label in tqdm(self.samples, desc="Caching"):
            img = self._load_single_image(path)
            if img is not None:
                cache.append((img, label))
        return cache

    def _load_single_image(self, path):
        try:
            meta = parse_phoenix_header(path)
            if meta is None: return None
            data = read_phoenix_data(path, meta) # Returns Complex Numpy Array
            
            # --- Preprocessing ---
            # 1. Log Magnitude Compression (Standard SAR processing)
            mag = np.abs(data)
            mag = 20 * np.log10(mag + 1e-6)
            # Min-Max Normalization to [0, 1]
            mag = (mag - mag.min()) / (mag.max() - mag.min())
            
            # 2. Reconstruct Complex form with normalized magnitude
            phase = np.angle(data)
            real = mag * np.cos(phase)
            imag = mag * np.sin(phase)
            
            # 3. Convert to Torch Tensor
            complex_data = torch.complex(torch.tensor(real, dtype=torch.float32), 
                                         torch.tensor(imag, dtype=torch.float32))
            
            # 4. Center Crop / Resize to 128x128
            # PhysX-MKS-GhostNet expects 128x128 input
            target_h, target_w = 128, 128
            h, w = complex_data.shape
            
            if h != target_h or w != target_w:
                # Interpolate Real and Imag separately
                complex_data = complex_data.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
                
                real_up = torch.nn.functional.interpolate(complex_data.real, size=(128,128), mode='bilinear', align_corners=False)
                imag_up = torch.nn.functional.interpolate(complex_data.imag, size=(128,128), mode='bilinear', align_corners=False)
                
                complex_data = torch.complex(real_up, imag_up).squeeze(0).squeeze(0)
            
            # Return (C, H, W) where C=1 (Complex channel)
            return complex_data.unsqueeze(0)
            
        except Exception:
            # Skip corrupt files gracefully
            return None

    def __len__(self):
        return len(self.data_cache) if self.cache_memory else len(self.samples)

    def __getitem__(self, idx):
        if self.cache_memory:
            img, label = self.data_cache[idx]
        else:
            path, label = self.samples[idx]
            img = self._load_single_image(path)
            if img is None: 
                # Fallback to next image if loading fails
                return self.__getitem__((idx + 1) % len(self))
        
        # Apply Augmentations (Transforms)
        if self.transform:
            img = self.transform(img)
            
        return img, labelnext