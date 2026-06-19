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


def Overlap(*hypervectors: ArrayLike, axis: Optional[int] = None):
    """Overlap similarity for sparse binary vectors, normalized by nonzeros in B.

    Counts the elements where both A[i] and B[i] are 1, normalized to the number
    of nonzero elements in the reference (second) hypervector B and mapped to
    [-1, 1] where 1 is identical, -1 is no overlap, and 0 is 50% overlap.
    Normalizing against B means the best results come from passing the bundled
    hypervector as A and the reference as B.

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
