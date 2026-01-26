import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.getcwd())
from src.models.net_architecture import PhysX_MKS_GhostNet

# --- CONFIG ---
DEVICE = torch.device("cpu")
CHECKPOINT_PATH = "outputs/checkpoints/best_model.pth"
ONNX_PATH = "outputs/physx_ghost.onnx"

class OnnxExportWrapper(nn.Module):
    """
    Final Wrapper: Forces PyTorch to handle complex math as real math 
    during ONNX serialization.
    """
    def __init__(self, model):
        super(OnnxExportWrapper, self).__init__()
        self.model = model

    def forward(self, x_real_view):
        # Input x_real_view comes in as (Batch, 1, 128, 128, 2) Float
        # Convert back to complex for the internal model
        x_complex = torch.view_as_complex(x_real_view)
        
        logits, recon, scatter, vlm = self.model(x_complex)
        
        # Convert complex outputs back to real views for ONNX compatibility
        recon_real_view = torch.view_as_real(recon)
        
        return logits, recon_real_view, scatter

def main():
    print("--- Exporting PhysX-MKS-GhostNet (Real-View Mode) ---")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint {CHECKPOINT_PATH} not found!")
        return

    # 1. Load Core Model
    core_model = PhysX_MKS_GhostNet(num_classes=10, width_mult=1.4).to(DEVICE)
    core_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    core_model.eval()
    
    # 2. Wrap for Export
    model = OnnxExportWrapper(core_model)
    
    # 3. Create Dummy Input using view_as_real
    # Original input is (1, 1, 128, 128) Complex
    dummy_complex = torch.randn(1, 1, 128, 128, dtype=torch.complex64)
    dummy_real_view = torch.view_as_real(dummy_complex) # Result: (1, 1, 128, 128, 2)
    
    print(f"Exporting to {ONNX_PATH}...")
    try:
        # We use a lower opset (11 or 12) to avoid complex-type auto-detection
        torch.onnx.export(
            model,
            dummy_real_view,
            ONNX_PATH,
            export_params=True,
            opset_version=11, 
            do_constant_folding=True,
            input_names=['input_view'],
            output_names=['logits', 'recon_view', 'scatter'],
            dynamic_axes={'input_view': {0: 'batch_size'}}
        )
        print("✅ SUCCESS: ONNX model exported with Real-View mapping.")
        
    except Exception as e:
        print(f"❌ Export Failed. Error: {e}")

if __name__ == "__main__":
    main()