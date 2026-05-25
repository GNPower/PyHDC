import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _normalize_inputs
from pyhdc.types import ArrayLike


def CosineSimilarity(*hypervectors: ArrayLike):
    """CosineSimilarity Cosine Similarity of two vectors

    cos(theta) = ( A dot B ) / ( norm(A) * norm(B) )

    Supports three calling conventions:
        (a, b) where a and b are 1D: returns a scalar in [-1, 1]
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
            a_t, b_t = torch.as_tensor(a).float(), torch.as_tensor(b).float()
            if a_t.ndim == 2:
                dots = (a_t * b_t).sum(dim=1)
                norms = torch.linalg.norm(a_t, dim=1) * torch.linalg.norm(b_t, dim=1)
                return dots / norms
            return (torch.dot(a_t, b_t) / (torch.linalg.norm(a_t) * torch.linalg.norm(b_t))).item()
        else:
            a_n, b_n = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
            if a_n.ndim == 2:
                dots = np.sum(a_n * b_n, axis=1)
                norms = np.linalg.norm(a_n, axis=1) * np.linalg.norm(b_n, axis=1)
                return dots / norms
            return np.dot(a_n, b_n) / (np.linalg.norm(a_n) * np.linalg.norm(b_n))
    else:
        arr = hvs[0]
        ref, rest = arr[0], arr[1:]
        if is_torch:
            assert torch is not None
            rf = torch.as_tensor(ref).float()
            rs = torch.as_tensor(rest).float()
            dots = rs @ rf
            norms = torch.linalg.norm(rs, dim=1) * torch.linalg.norm(rf)
            return dots / norms
        else:
            rf = np.asarray(ref, dtype=np.float32)
            rs = np.asarray(rest, dtype=np.float32)
            return (rs @ rf) / (np.linalg.norm(rs, axis=1) * np.linalg.norm(rf))
