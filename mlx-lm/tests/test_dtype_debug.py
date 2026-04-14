import mlx.core as mx
import numpy as np

# Simulate what pack_indices might return
x = mx.array([1, 2, 3], dtype=mx.uint8)
print(f"dtype of x: {x.dtype}")
print(f"type of x.dtype: {type(x.dtype)}")
try:
    z = mx.zeros((2, 2), dtype=x.dtype)
    print("mx.zeros works with x.dtype")
except Exception as e:
    print(f"Error: {e}")
