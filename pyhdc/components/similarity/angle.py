import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

def AngleDistance(a, b):
    """Average angular distance of two complex unit hypervectors, normalized to [0, 1]."""
    if TORCH_AVAILABLE and torch.is_tensor(a):
        return (torch.cos(a - b).sum() / a.numel()).item()
    else:
        return np.sum(np.cos(a - b)) / a.size
