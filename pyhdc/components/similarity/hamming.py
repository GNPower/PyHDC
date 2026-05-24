import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

def HammingDistance(a, b):
    """HammingDistance Hamming Distance of two vectors

    Counts the number of elements in the hypervectors where A[i] != B[i].
    Result is normalized to the size of the hypervector so it is always in
    the range [0,1].

    Args:
        a: hypervector A
        b: hypervector B

    Returns:
        Similarity between hypervectors
    """
    if TORCH_AVAILABLE and torch.is_tensor(a):
        return 1 - ((a != b).sum().item() / a.numel())
    else:
        return 1 - (np.count_nonzero(a != b) / a.size)
