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


def AngleDistance(
    *hypervectors: ArrayLike, axis: Optional[int] = None, mode: str = "pairwise"
):
    """AngleDistance Angle Distance of phase hypervectors

    Calculates the average angular distance of two complex hypervectors of unit
    length, normalized to the hypervector dimension so it is in [-1, 1].

    Hypervectors are dimension-first (axis 0 is always the dimension ``D``).
    Supports these calling conventions::

        (a, b) where a and b are 1D: returns a scalar in [-1, 1]
        (a, b) batches:              per-pair scores (trailing axes broadcast)
        (arr,) where arr is (D, N):  sim(col_0, col_i) for i in 1..N-1
        (arr,) where arr is (D, N, M, ...): requires ``axis``

    With ``mode="cross"`` and two phase batches ``A=(D, P)``, ``B=(D, M)``, returns
    the full ``(P, M)`` matrix using ``cos(x - y) = cos x cos y + sin x sin y`` so
    the reduction over ``D`` is two matmuls and no ``(D, P, M)`` tensor is built.

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
            dimension = a_t.shape[0]
            return (
                torch.cos(a_t).T @ torch.cos(b_t) + torch.sin(a_t).T @ torch.sin(b_t)
            ) / dimension
        a_n = np.asarray(a, dtype=np.float64)
        b_n = np.asarray(b, dtype=np.float64)
        dimension = a_n.shape[0]
        return (np.cos(a_n).T @ np.cos(b_n) + np.sin(a_n).T @ np.sin(b_n)) / dimension

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
