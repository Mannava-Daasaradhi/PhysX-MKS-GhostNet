import torch
import torch.nn as nn

class PhysXLoss(nn.Module):
    """
    PhysX-MKS-GhostNet Loss Function.
    Combines Classification, Reconstruction, and Physics Constraints.
    
    Formula: L_total = L_cls + (alpha * L_recon) + (beta * L_phys)
    """
    def __init__(self, alpha=0.2, beta=0.05): 
        """
        TUNED WEIGHTS for 99% Convergence:
        alpha: 0.2  (Reconstruction - keeps image clean)
        beta:  0.05 (Physics Sparsity - forces geometric feature learning)
        """
        super(PhysXLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, recon, scatter, targets, inputs):
        """
        Args:
            logits:  (B, Num_Classes) - Classification output
            recon:   (B, 1, H, W) - Reconstructed Complex Image
            scatter: (B, 1, H, W) - Physics Scattering Map (0-1)
            targets: (B) - Ground Truth Labels
            inputs:  (B, 1, H, W) - Original Input Image (Complex)
        """
        # 1. Classification Loss (The Primary Goal)
        loss_cls = self.ce_loss(logits, targets)
        
        # 2. Reconstruction Loss (Self-Supervision)
        # Ensure dimensions match (handling potential rounding in downsampling)
        if recon.shape != inputs.shape:
             recon = torch.nn.functional.interpolate(
                 recon.real, size=inputs.shape[2:], mode='bilinear'
             ) + 1j * torch.nn.functional.interpolate(
                 recon.imag, size=inputs.shape[2:], mode='bilinear'
             )

        # Complex MSE: Average of Real^2 and Imag^2 errors
        diff = recon - inputs
        loss_recon = torch.mean(diff.real**2 + diff.imag**2)
        
        # 3. Physics Sparsity Loss (The "Trust" Constraint)
        # We want the scattering map to be sparse (mostly zero, except for corners).
        # L1 Norm encourages sparsity.
        loss_phys = torch.mean(torch.abs(scatter))
        
        # Total Weighted Loss
        total_loss = loss_cls + (self.alpha * loss_recon) + (self.beta * loss_phys)
        
        return total_loss