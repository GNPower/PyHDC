import numpy as np

# Optional PyTorch support
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _normalize_inputs

# Type aliases
from pyhdc.types import ArrayLike

# ============================================================================
# Binary Operations
# ============================================================================


def Disjunction(*hypervectors: ArrayLike) -> ArrayLike:
    """
    Bitwise OR bundling for sparse binary vectors.

    Bundles sparse binary hypervectors using bitwise OR. An element is 1
    in the result if it is 1 in any input vector. Preserves sparsity better
    than addition for Binary Sparse Distributed Codes.

    Args:
        *hypervectors: Variable number of sparse binary hypervectors, or single 2D batch

    Returns:
        Bundled sparse binary hypervector

    Example:
        >>> v1 = np.array([1, 0, 1, 0])
        >>> v2 = np.array([0, 1, 0, 0])
        >>> result = Disjunction(v1, v2)
        >>> # result: [1, 1, 1, 0]
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)

    if is_torch:
        result = hvs[0].bool()
        for i in range(1, len(hvs)):
            result = torch.logical_or(result, hvs[i].bool())
        return result.to(hvs[0].dtype)
    else:
        return np.bitwise_or.reduce(hvs).astype(hvs[0].dtype)
