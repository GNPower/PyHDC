import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _normalize_inputs
from pyhdc.types import ArrayLike


def HammingDistance(*hypervectors: ArrayLike):
    """HammingDistance Hamming Distance of two vectors

    Counts the number of elements in the hypervectors where A[i] != B[i].
    Result is normalized to the size of the hypervector and mapped to [-1, 1]
    where 1 is identical, -1 is completely different, and 0 is half-matching.

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
            if a_t.ndim == 2:
                mismatches = (a_t != b_t).int().sum(dim=1).float()
                return 1 - 2 * mismatches / a_t.shape[1]
            return 1 - 2 * int((a_t != b_t).sum()) / a_t.numel()
        else:
            a_n, b_n = np.asarray(a), np.asarray(b)
            if a_n.ndim == 2:
                mismatches = np.not_equal(a_n, b_n).sum(axis=1)
                return 1 - 2 * mismatches / a_n.shape[1]
            return 1 - 2 * np.count_nonzero(np.not_equal(a_n, b_n)) / a_n.size
    else:
        arr = hvs[0]
        ref, rest = arr[0], arr[1:]
        if is_torch:
            assert torch is not None
            rest_t, ref_t = torch.as_tensor(rest), torch.as_tensor(ref)
            mismatches = (rest_t != ref_t).int().sum(dim=1).float()
            return 1 - 2 * mismatches / ref_t.numel()
        else:
            rest_n, ref_n = np.asarray(rest), np.asarray(ref)
            mismatches = np.not_equal(rest_n, ref_n).sum(axis=1)
            return 1 - 2 * mismatches / ref_n.size
