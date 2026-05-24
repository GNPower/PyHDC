import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

def AngleDistance(a, b):
    """AngleDistance Angle Distance of two hypervectors

    Calculates the average angular distance of two complex hypervectors
    of unit length. Results are normalized to the size of the hypervectors
    so it is always in the range [0,1].

    Args:
        a: hypervector A
        b: hypervector B

    Returns:
        Similarity between hypervectors
    """
    if TORCH_AVAILABLE and torch.is_tensor(a):
        return (torch.cos(a - b).sum() / a.numel()).item()
    else:
        return np.sum(np.cos(a - b)) / a.size
