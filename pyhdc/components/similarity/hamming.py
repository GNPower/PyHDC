import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

def HammingDistance(a, b):
    """Normalized Hamming similarity: fraction of matching elements, in [0, 1]."""
    if TORCH_AVAILABLE and torch.is_tensor(a):
        return 1 - ((a != b).sum().item() / a.numel())
    else:
        return 1 - (np.count_nonzero(a != b) / a.size)
