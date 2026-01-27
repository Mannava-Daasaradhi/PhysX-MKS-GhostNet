import numpy as np
import onnxruntime as ort
import time
import os
import argparse

def benchmark_config(model_path, threads, mode_name):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    try:
        sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Skipping {mode_name}: {e}")
        return 0

    # Dummy Input
    input_node = sess.get_inputs()[0]
    input_shape = (1, 2, 128, 128)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(20):
        sess.run(None, {input_node.name: dummy_input})

    # Run
    start = time.time()
    for _ in range(200):
        sess.run(None, {input_node.name: dummy_input})
    end = time.time()

    fps = 200.0 / (end - start)
    print(f"  [{mode_name}] Threads={threads} :: FPS = {fps:.2f}")
    return fps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='outputs/physx_ghost.onnx')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print("Model not found!")
        return

    print(f"--- Tuning PhysX-MKS-GhostNet on Raspberry Pi 5 ---")
    print(f"Model: {args.model}")
    
    # Run Sweep
    res_1 = benchmark_config(args.model, 1, "Single-Core")
    res_2 = benchmark_config(args.model, 2, "Dual-Core  ")
    res_4 = benchmark_config(args.model, 4, "Quad-Core  ")

    print(f"\n🏆 WINNER: {max(res_1, res_2, res_4):.2f} FPS")

if __name__ == "__main__":
    main()