import torch
import torch.nn as nn
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os
import sys

# Add src to path just in case
sys.path.append(os.getcwd())

from src.models.net_architecture import PhysX_MKS_GhostNet

# --- CONFIG ---
DEVICE = torch.device("cpu") 
CHECKPOINT_PATH = "outputs/checkpoints/best_model.pth"
ONNX_PATH = "outputs/physx_ghost.onnx"
TF_PATH = "outputs/physx_ghost_tf"
TFLITE_PATH = "outputs/physx_ghost.tflite"

def main():
    print("--- Starting PhysX-MKS-GhostNet Conversion ---")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint {CHECKPOINT_PATH} not found!")
        return

    # 1. Load PyTorch Model
    print(f"Loading model...")
    model = PhysX_MKS_GhostNet(num_classes=10, width_mult=1.4).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # 2. Export to ONNX
    dummy_input = torch.randn(1, 1, 128, 128, dtype=torch.complex64).to(DEVICE)
    
    print(f"Exporting to ONNX...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_PATH,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['logits', 'recon', 'scatter', 'vlm'],
            dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
        )
    except Exception as e:
        print(f"ONNX Export Error: {e}")
        return

    # 3. ONNX -> TensorFlow
    print("Converting to TF...")
    onnx_model = onnx.load(ONNX_PATH)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(TF_PATH)

    # 4. TF -> TFLite
    print(f"Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)
        
    print(f"SUCCESS: Model saved to {TFLITE_PATH}")

if __name__ == "__main__":
    main()