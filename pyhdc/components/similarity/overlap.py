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


def Overlap(
    *hypervectors: ArrayLike, axis: Optional[int] = None, mode: str = "pairwise"
):
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

    With ``mode="cross"`` and two sparse binary ``{0, 1}`` batches ``A=(D, P)``
    (prototypes), ``B=(D, M)`` (codebook), returns the full ``(P, M)`` matrix. The
    normalization stays asymmetric, each column ``m`` is divided by the nonzero
    count of ``B[:, m]`` (the second argument). So pass the bundled vectors as A
    and the reference codebook as B.

    Args:
        *hypervectors: Two hypervectors, or a single batch array
        axis: For a single ``(D, N, M, ...)`` batch, the batch axis to split on
        mode: ``"pairwise"`` (default) or ``"cross"``

    Returns:
        Scalar similarity, or an array of similarities over the trailing axes
    """
    a, b, is_torch, scalar = _normalize_similarity(*hypervectors, axis=axis, mode=mode)

    if mode == "cross":
        if is_torch:
            assert torch is not None
            a_t = torch.as_tensor(a).double()
            b_t = torch.as_tensor(b).double()
            shared = a_t.T @ b_t
            denom = b_t.sum(dim=0).clamp(min=1)[None, :]
            return 2 * shared / denom - 1
        # Inputs are cast to float64 for a BLAS matmul
        a_n = np.asarray(a, dtype=np.float64)
        b_n = np.asarray(b, dtype=np.float64)
        shared = a_n.T @ b_n
        denom = np.maximum(b_n.sum(axis=0), 1)[None, :]
        return 2 * shared / denom - 1

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
