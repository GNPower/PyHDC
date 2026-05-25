import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _normalize_inputs
from pyhdc.types import ArrayLike


def Overlap(*hypervectors: ArrayLike):
    """Overlap similarity for sparse binary vectors, normalized by nonzeros in b, in [0, 1].

    Counts the number of elements in the hypervectors where both A[i] and
    B[i] are 1. Useful for sparse hypervectors. Result is normalized to the
    number of nonzero elements in the reference (second) hypervector and
    mapped to [-1, 1] where 1 is identical, -1 is no overlap, and 0 is
    50% overlap. Normalizing against the second hypervector means the best
    results will be achieved from passing the bundled hypervector as A and
    the reference hypervector as B.

    Supports three calling conventions:
        (a, b) where a and b are 1D: returns a scalar in [0, 1]
        (a, b) where a and b are 2D: returns a 1D array of per-row scalars
        (arr,) where arr is 2D:      returns a 1D array of sim(row_0, row_i)
                                     for i in 1..N-1

    Args:
        *hypervectors: Two 1D/2D hypervectors, or a single 2D array of hypervectors

    Returns:
        Scalar similarity, or 1D array of similarities
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)

    if len(hvs) == 2:
        a, b = hvs[0], hvs[1]
        if is_torch:
            assert torch is not None
            a_t, b_t = torch.as_tensor(a), torch.as_tensor(b)
            mask = torch.logical_and(a_t == b_t, a_t == 1)
            if a_t.ndim == 2:
                return (
                    2 * mask.sum(dim=1).float() / b_t.sum(dim=1).float().clamp(min=1)
                    - 1
                )
            return 2 * mask.sum().item() / max(b_t.sum().item(), 1) - 1
        else:
            a_n, b_n = np.asarray(a), np.asarray(b)
            mask = np.logical_and(a_n == b_n, a_n == 1)
            if a_n.ndim == 2:
                return 2 * mask.sum(axis=1) / np.maximum(b_n.sum(axis=1), 1) - 1
            return 2 * int(mask.sum()) / max(int(b_n.sum()), 1) - 1
    else:
        arr = hvs[0]
        ref, rest = arr[0], arr[1:]
        if is_torch:
            assert torch is not None
            rest_t, ref_t = torch.as_tensor(rest), torch.as_tensor(ref)
            mask = torch.logical_and(rest_t == ref_t, rest_t == 1)
            return 2 * mask.sum(dim=1).float() / max(ref_t.sum().item(), 1) - 1
        else:
            rest_n, ref_n = np.asarray(rest), np.asarray(ref)
            mask = np.logical_and(rest_n == ref_n, rest_n == 1)
            return 2 * mask.sum(axis=1) / max(int(ref_n.sum()), 1) - 1
