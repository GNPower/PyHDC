import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

def Overlap(a, b):
    """Overlap similarity for sparse binary vectors, normalized by nonzeros in b, in [0, 1].

    Counts the number of elements in the hyypervectors where both A[i] and
    B[i] are 1. Useful for sparce hypervectors. Result is normalized to the
    number of nonzero elements in the reference (second) hypervector so it
    is always in the range [0,1]. Normalizing against the second hypervector
    means the best results will be acheived from passing the bundled
    hypervector as A and the reference hypervector as B.

    Args:
        a: hypervector A
        b: hypervector B

    Returns:
        Similarity between hypervectors
    """
    if TORCH_AVAILABLE and torch.is_tensor(a):
        intersection = torch.logical_and(a == b, a == 1).sum().item()
        return intersection / max(b.sum().item(), 1)
    else:
        return np.count_nonzero(np.logical_and(a == b, a == 1)) / max(np.sum(b), 1)
