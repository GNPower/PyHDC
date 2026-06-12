import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _normalize_similarity
from pyhdc.types import ArrayLike


def HammingDistance(*hypervectors: ArrayLike):
    """HammingDistance Hamming Distance of hypervectors

    Counts the number of elements where A[i] != B[i], normalized to the
    hypervector dimension and mapped to [-1, 1] where 1 is identical, -1 is
    completely different, and 0 is half-matching.

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
        a_t = torch.as_tensor(a)
        b_t = torch.as_tensor(b)
        dimension = a_t.shape[0]
        mismatches = (a_t != b_t).sum(dim=0).float()
        sims = 1 - 2 * mismatches / dimension
        return sims.item() if scalar else sims

    a_n = np.asarray(a)
    b_n = np.asarray(b)
    dimension = a_n.shape[0]
    mismatches = np.not_equal(a_n, b_n).sum(axis=0)
    sims = 1 - 2 * mismatches / dimension
    return sims.item() if scalar else sims
