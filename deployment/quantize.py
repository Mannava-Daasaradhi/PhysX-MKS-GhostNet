import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def main():
    input_model = "outputs/physx_ghost.onnx"
    output_model = "outputs/physx_ghost_int8.onnx"

    if not os.path.exists(input_model):
        print(f"Error: {input_model} not found.")
        return

    print(f"Quantizing {input_model}...")
    quantize_dynamic(
        input_model, 
        output_model, 
        weight_type=QuantType.QUInt8
    )
    print(f"Success! Saved to {output_model}")

if __name__ == "__main__":
    main()