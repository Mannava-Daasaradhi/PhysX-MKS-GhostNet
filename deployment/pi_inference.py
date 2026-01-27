import numpy as np
import onnxruntime as ort
import time
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='physx_ghost.onnx', help='Path to ONNX model')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model {args.model} not found.")
        return

    # 1. Initialize Session
    # Try CUDA (Jetson) first, fall back to CPU (Pi)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        sess = ort.InferenceSession(args.model, providers=providers)
    except Exception as e:
        print(f"Warning: Could not load CUDA provider. Fallback to CPU. ({e})")
        sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])

    print(f"--- Running on: {sess.get_providers()[0]} ---")

    # 2. Get Input Info
    input_node = sess.get_inputs()[0]
    input_name = input_node.name
    
    # Force expected shape for Shadow Mode: (Batch, 2, 128, 128)
    # Channel 0 = Real, Channel 1 = Imaginary
    input_shape = (1, 2, 128, 128)
    print(f"Input Shape: {input_shape}")

    # 3. Create Dummy Data
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # 4. Warmup
    print("Warming up...")
    for _ in range(10):
        sess.run(None, {input_name: dummy_input})

    # 5. Benchmark
    print("Benchmarking (100 runs)...")
    latencies = []
    
    start_total = time.time()
    for _ in range(100):
        t0 = time.time()
        sess.run(None, {input_name: dummy_input})
        latencies.append((time.time() - t0) * 1000) # ms
    end_total = time.time()

    # 6. Report
    avg_ms = np.mean(latencies)
    fps = 1000.0 / avg_ms
    print(f"\nRESULTS:")
    print(f"Avg Latency: {avg_ms:.2f} ms")
    print(f"Throughput:  {fps:.2f} FPS")

if __name__ == "__main__":
    main()