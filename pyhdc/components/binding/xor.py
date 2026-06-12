import numpy as np

# Optional PyTorch support
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _normalize_binding

# Type aliases
from pyhdc.types import ArrayLike

# ============================================================================
# XOR-based Binding
# ============================================================================


def ExclusiveOr(*hypervectors: ArrayLike) -> ArrayLike:
    """
    XOR binding for binary hypervectors.

    Binds binary hypervectors using exclusive OR. Used with Binary Spatter
    Codes (BSC) and other binary encodings. XOR is its own inverse.

    Args:
        *hypervectors: Variable number of binary hypervectors, or single 2D batch

    Returns:
        Bound binary hypervector

    Example:
        >>> v1 = np.array([1, 0, 1, 0])
        >>> v2 = np.array([1, 1, 0, 0])
        >>> result = ExclusiveOr(v1, v2)
        >>> # result: [0, 1, 1, 0]
    """
    hvs, is_torch, _ = _normalize_binding(*hypervectors)

    if is_torch:
        result = hvs[0].bool()
        for i in range(1, len(hvs)):
            result = torch.logical_xor(result, hvs[i].bool())
        return result.to(hvs[0].dtype)
    else:
        return np.logical_xor.reduce(hvs).astype(hvs[0].dtype)
