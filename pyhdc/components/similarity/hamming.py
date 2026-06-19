from typing import Optional

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _normalize_similarity
from pyhdc.types import ArrayLike


def HammingDistance(*hypervectors: ArrayLike, axis: Optional[int] = None):
    """HammingDistance Hamming Distance of hypervectors

    Counts the number of elements where A[i] != B[i], normalized to the
    hypervector dimension and mapped to [-1, 1] where 1 is identical, -1 is
    completely different, and 0 is half-matching.

    Hypervectors are dimension-first (axis 0 is always the dimension ``D``).
    Supports these calling conventions::

        (a, b) where a and b are 1D: returns a scalar in [-1, 1]
        (a, b) batches:              per-pair scores (trailing axes broadcast)
        (arr,) where arr is (D, N):  sim(col_0, col_i) for i in 1..N-1
        (arr,) where arr is (D, N, M, ...): requires ``axis``

    Args:
        *hypervectors: Two hypervectors, or a single batch array
        axis: For a single ``(D, N, M, ...)`` batch, the batch axis to split on

    Returns:
        Scalar similarity, or an array of similarities over the trailing axes
    """
    a, b, is_torch, scalar = _normalize_similarity(*hypervectors, axis=axis)

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
