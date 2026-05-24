import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

def Overlap(a, b):
    """Overlap similarity for sparse binary vectors, normalized by nonzeros in b, in [0, 1].

    Pass the bundled vector as a and the reference vector as b for best results.
    """
    if TORCH_AVAILABLE and torch.is_tensor(a):
        intersection = torch.logical_and(a == b, a == 1).sum().item()
        return intersection / max(b.sum().item(), 1)
    else:
        return np.count_nonzero(np.logical_and(a == b, a == 1)) / max(np.sum(b), 1)
