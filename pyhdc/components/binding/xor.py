import numpy as np

# Optional PyTorch support
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _broadcast_operands, _normalize_binding

# Type aliases
from pyhdc.types import ArrayLike

# ============================================================================
# XOR-based Binding
# ============================================================================


def ExclusiveOr(*hypervectors: ArrayLike) -> ArrayLike:
    """
    XOR binding for binary hypervectors.

    Binds binary hypervectors using exclusive OR. Used with Binary Spatter
    Codes (BSC) and other binary encodings. XOR is its own inverse. Operands
    broadcast over the trailing batch axes (a single key binds against each
    column of a batch, two batches bind per column).

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
    operands = _broadcast_operands(hvs, is_torch)

    if is_torch:
        result = operands[0].bool()
        for operand in operands[1:]:
            result = torch.logical_xor(result, operand.bool())
        return result.to(operands[0].dtype)

    result = operands[0].astype(bool)
    for operand in operands[1:]:
        result = np.logical_xor(result, operand.astype(bool))
    return result.astype(operands[0].dtype)
