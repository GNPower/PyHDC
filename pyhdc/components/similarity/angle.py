import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _normalize_similarity
from pyhdc.types import ArrayLike


def AngleDistance(*hypervectors: ArrayLike):
    """AngleDistance Angle Distance of phase hypervectors

    Calculates the average angular distance of two complex hypervectors of unit
    length, normalized to the hypervector dimension so it is in [-1, 1].

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
        sims = torch.cos(a_t - b_t).sum(dim=0) / dimension
        return sims.item() if scalar else sims

    a_n = np.asarray(a)
    b_n = np.asarray(b)
    dimension = a_n.shape[0]
    sims = np.cos(a_n - b_n).sum(axis=0) / dimension
    return sims.item() if scalar else sims
