import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _normalize_similarity
from pyhdc.types import ArrayLike


def Overlap(*hypervectors: ArrayLike):
    """Overlap similarity for sparse binary vectors, normalized by nonzeros in B.

    Counts the elements where both A[i] and B[i] are 1, normalized to the number
    of nonzero elements in the reference (second) hypervector B and mapped to
    [-1, 1] where 1 is identical, -1 is no overlap, and 0 is 50% overlap.
    Normalizing against B means the best results come from passing the bundled
    hypervector as A and the reference as B.

    Hypervectors are dimension-first ``(D, N)`` (each column is a hypervector).
    Supports three calling conventions:
        (a, b) where a and b are 1D: returns a scalar in [-1, 1]
        (a, b) where a and b are (D, N): returns a 1D array of per-column scores
        (arr,) where arr is (D, N):     returns a 1D array of sim(col_0, col_i)
                                        for i in 1..N-1, normalized by each col_i

    Args:
        *hypervectors: Two 1D/2D hypervectors, or a single (D, N) array

    Returns:
        Scalar similarity, or 1D array of similarities
    """
    a, b, is_torch, scalar = _normalize_similarity(*hypervectors)

    if is_torch:
        assert torch is not None
        a_t = torch.as_tensor(a)
        b_t = torch.as_tensor(b)
        mask = torch.logical_and(a_t == b_t, a_t == 1)
        denom = b_t.sum(dim=0).float().clamp(min=1)
        sims = 2 * mask.sum(dim=0).float() / denom - 1
        return sims.item() if scalar else sims

    a_n = np.asarray(a)
    b_n = np.asarray(b)
    mask = np.logical_and(a_n == b_n, a_n == 1)
    denom = np.maximum(b_n.sum(axis=0), 1)
    sims = 2 * mask.sum(axis=0) / denom - 1
    return sims.item() if scalar else sims
