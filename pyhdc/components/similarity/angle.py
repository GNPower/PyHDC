import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _normalize_inputs
from pyhdc.types import ArrayLike


def AngleDistance(*hypervectors: ArrayLike):
    """AngleDistance Angle Distance of two hypervectors

    Calculates the average angular distance of two complex hypervectors
    of unit length. Results are normalized to the size of the hypervectors
    so it is always in the range [0,1].

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
                return torch.cos(a_t - b_t).sum(dim=1) / a_t.shape[1]
            return (torch.cos(a_t - b_t).sum() / a_t.numel()).item()
        else:
            a_n, b_n = np.asarray(a), np.asarray(b)
            if a_n.ndim == 2:
                return np.cos(a_n - b_n).sum(axis=1) / a_n.shape[1]
            return float(np.sum(np.cos(a_n - b_n)) / a_n.size)
    else:
        arr = hvs[0]
        ref, rest = arr[0], arr[1:]
        if is_torch:
            assert torch is not None
            rest_t, ref_t = torch.as_tensor(rest), torch.as_tensor(ref)
            return torch.cos(rest_t - ref_t).sum(dim=1) / ref_t.numel()
        else:
            rest_n, ref_n = np.asarray(rest), np.asarray(ref)
            return np.cos(rest_n - ref_n).sum(axis=1) / ref_n.size
