import mlx.core as mx
import time
import numpy as np
import json
import argparse
import os

def calibrate_roofline():
    # Peak FP16 TFLOPS
    # Large GEMM to saturate compute
    N = 8192
    a = mx.random.uniform(shape=(N, N)).astype(mx.float16)
    b = mx.random.uniform(shape=(N, N)).astype(mx.float16)
    
    # Warmup
    for _ in range(5):
        mx.eval(mx.matmul(a, b))
    
    start = time.perf_counter()
    iters = 10
    for _ in range(iters):
        mx.eval(mx.matmul(a, b))
    end = time.perf_counter()
    
    t_avg = (end - start) / iters
    flops = 2 * (N ** 3)
    tflops = (flops / t_avg) / 1e12
    
    # Peak Bandwidth (Stream Triad)
    # read-write-read pattern: c = a + b * scalar
    S = 128 * 1024 * 1024  # 128M elements
    a_bw = mx.random.uniform(shape=(S,)).astype(mx.float32)
    b_bw = mx.random.uniform(shape=(S,)).astype(mx.float32)
    scalar = 1.5
    
    # Warmup
    for _ in range(5):
        mx.eval(a_bw + b_bw * scalar)
    
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(a_bw + b_bw * scalar)
    end = time.perf_counter()
    
    t_avg_bw = (end - start) / iters
    # 2 reads (a, b) + 1 write (c) = 3 * S * 4 bytes
    bytes_moved = 3 * S * 4
    gb_s = (bytes_moved / t_avg_bw) / 1e9
    
    # Dispatch Overhead
    def noop():
        pass
    
    start = time.perf_counter()
    for _ in range(1000):
        mx.eval(mx.array([0]))
    end = time.perf_counter()
    dispatch_us = ((end - start) / 1000) * 1e6

    return {
        "peak_fp16_tflops": round(tflops, 2),
        "peak_memory_bandwidth_gbs": round(gb_s, 2),
        "noop_dispatch_us": round(dispatch_us, 2)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/roofline_m4max.json")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = calibrate_roofline()
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Calibration complete: {results}")
