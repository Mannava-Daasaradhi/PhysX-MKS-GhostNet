import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def main():
    # --- CONFIG ---
    # Corrected paths relative to project root
    input_model_path = "outputs/physx_ghost.onnx"
    output_model_path = "outputs/physx_ghost_int8.onnx"

    print(f"--- Quantizing Model ---")
    print(f"Input:  {input_model_path}")
    print(f"Output: {output_model_path}")

    # 1. Verify Input Exists
    if not os.path.exists(input_model_path):
        print(f"\n❌ ERROR: File not found: {input_model_path}")
        print(f"   Please check if the 'outputs' folder exists.")
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
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")

if __name__ == "__main__":
    main()