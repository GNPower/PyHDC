import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _normalize_similarity
from pyhdc.types import ArrayLike


def CosineSimilarity(*hypervectors: ArrayLike):
    """CosineSimilarity Cosine Similarity of hypervectors

    cos(theta) = ( A dot B ) / ( norm(A) * norm(B) )

    Hypervectors are dimension-first ``(D, N)`` (each column is a hypervector).
    Supports three calling conventions:
        (a, b) where a and b are 1D: returns a scalar in [-1, 1]
        (a, b) where a and b are (D, N): returns a 1D array of per-column scores
        (arr,) where arr is (D, N):     returns a 1D array of sim(col_0, col_i)
                                        for i in 1..N-1

    Args:
        *hypervectors: Two 1D/2D hypervectors, or a single (D, N) array

    Returns:
        Scalar similarity, or 1D array of similarities
    """
    a, b, is_torch, scalar = _normalize_similarity(*hypervectors)

    if is_torch:
        assert torch is not None
        a_t = torch.as_tensor(a).float()
        b_t = torch.as_tensor(b).float()
        dots = (a_t * b_t).sum(dim=0)
        norms = torch.linalg.norm(a_t, dim=0) * torch.linalg.norm(b_t, dim=0)
        sims = dots / norms
        return sims.item() if scalar else sims

    a_n = np.asarray(a, dtype=np.float32)
    b_n = np.asarray(b, dtype=np.float32)
    dots = np.sum(a_n * b_n, axis=0)
    norms = np.linalg.norm(a_n, axis=0) * np.linalg.norm(b_n, axis=0)
    sims = dots / norms
    return sims.item() if scalar else sims
