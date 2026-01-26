import numpy as np
import time
import os
import argparse

# Try standard imports for Pi or Desktop
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='physx_ghost.tflite', help='Path to .tflite file')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model file '{args.model}' not found.")
        print("Please move 'physx_ghost.tflite' to this folder or specify path.")
        return

    print(f"--- Loading {args.model} on Edge Device ---")
    
    # 1. Load Model
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 2. Dummy Inference Loop
    input_shape = input_details[0]['shape']
    print(f"Input Shape: {input_shape}")
    
    # Create random complex-like data (float32 representation)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    print("Warming up...")
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    
    print("Running Benchmark (100 runs)...")
    times = []
    for _ in range(100):
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        times.append((time.time() - start) * 1000)
        
    avg_ms = np.mean(times)
    fps = 1000 / avg_ms
    print(f"RESULTS: {avg_ms:.2f} ms per frame | {fps:.2f} FPS")

if __name__ == "__main__":
    main()
