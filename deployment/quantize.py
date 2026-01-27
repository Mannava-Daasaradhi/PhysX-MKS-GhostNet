import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import sys

def main():
    # --- CONFIG ---
    # The path relative to where you run the script (project root)
    input_model_path = "outputs/physx_ghost.onnx"
    output_model_path = "outputs/physx_ghost_int8.onnx"

    print(f"--- Quantizing Model ---")
    print(f"Input:  {input_model_path}")
    print(f"Output: {output_model_path}")

    # 1. Verify Input Exists
    if not os.path.exists(input_model_path):
        print(f"\n❌ ERROR: File not found: {input_model_path}")
        print(f"   Did you run export_to_onnx.py? Check if the 'outputs' folder exists.")
        return

    # 2. Run Quantization
    try:
        quantize_dynamic(
            model_input=input_model_path,
            model_output=output_model_path,
            weight_type=QuantType.QUInt8 
        )
        print(f"\n✅ SUCCESS: Quantization complete!")
        print(f"   Saved to: {output_model_path}")
        print("   You can now transfer this INT8 model to your Raspberry Pi.")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")

if __name__ == "__main__":
    main()