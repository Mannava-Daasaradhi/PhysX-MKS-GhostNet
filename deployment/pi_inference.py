import numpy as np
import onnxruntime as ort
import time

def main():
    model_path = 'physx_ghost.onnx'
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    
    # Create input: (1, 1, 128, 128, 2) Float32
    # The last dimension [:, :, :, :, 0] is Real, [:, :, :, :, 1] is Imag
    dummy_input = np.random.randn(1, 1, 128, 128, 2).astype(np.float32)
    
    print("Running Raspberry Pi Benchmark...")
    start = time.time()
    for _ in range(100):
        outputs = sess.run(None, {input_name: dummy_input})
    end = time.time()
    
    print(f"Avg Latency: {(end - start) * 10:.2f} ms")

if __name__ == "__main__":
    main()