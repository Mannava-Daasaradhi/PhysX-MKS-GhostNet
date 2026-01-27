import numpy as np
import onnxruntime as ort
import time
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    # Default to the Float32 model (it was faster)
    parser.add_argument('--model', type=str, default='outputs/physx_ghost.onnx', help='Path to ONNX model')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model {args.model} not found.")
        return

    # --- PERFORMANCE TUNING ---
    sess_options = ort.SessionOptions()
    # Raspberry Pi 4 has 4 physical cores. Setting this to 4 usually gives max speed.
    sess_options.intra_op_num_threads = 4
    # Enable all internal optimizations (Constant Folding, etc.)
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Sequential execution is often faster for small batches
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Initialize Session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        sess = ort.InferenceSession(args.model, sess_options=sess_options, providers=providers)
    except:
        sess = ort.InferenceSession(args.model, sess_options=sess_options, providers=['CPUExecutionProvider'])

    print(f"--- Running on: {sess.get_providers()[0]} ---")
    print(f"--- Threads: {sess_options.intra_op_num_threads} ---")

    # Get Input Info
    input_node = sess.get_inputs()[0]
    input_name = input_node.name
    # Force Shape: (Batch, 2, 128, 128) -> [Real, Imag]
    input_shape = (1, 2, 128, 128)
    
    # Create Dummy Data
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    print("Warming up...")
    for _ in range(20):
        sess.run(None, {input_name: dummy_input})

    # Benchmark
    print("Benchmarking (200 runs)...")
    latencies = []
    
    for _ in range(200):
        t0 = time.time()
        sess.run(None, {input_name: dummy_input})
        latencies.append((time.time() - t0) * 1000) # ms

    # Report
    avg_ms = np.mean(latencies)
    fps = 1000.0 / avg_ms
    print(f"\nRESULTS (Float32 Optimized):")
    print(f"Avg Latency: {avg_ms:.2f} ms")
    print(f"Throughput:  {fps:.2f} FPS")

if __name__ == "__main__":
    main()