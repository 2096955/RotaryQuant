import mlx.core as mx
import numpy as np
from mlx_lm.models.mlx_turboquant import pack_indices

indices_mx = mx.zeros((1, 2, 10, 128), dtype=mx.uint8)
packed_k = pack_indices(indices_mx)
print(f"type(packed_k): {type(packed_k)}")
if hasattr(packed_k, 'dtype'):
    print(f"packed_k.dtype: {packed_k.dtype}")
    print(f"type(packed_k.dtype): {type(packed_k.dtype)}")
