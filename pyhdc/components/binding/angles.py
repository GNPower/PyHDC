from math import pi

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
# Angle-based Binding (for FHRR)
# ============================================================================


def ElementAngleAddition(*hypervectors: ArrayLike) -> ArrayLike:
    """
    Binding by adding phase angles.

    Used with Fourier Holographic Reduced Representations (FHRR) where
    hypervector elements represent phase angles. Operands broadcast over the
    trailing batch axes.

    Args:
        *hypervectors: Variable number of phase hypervectors, or single 2D batch

    Returns:
        Bound hypervector with phases in [0, 2*pi)
    """
    hvs, is_torch, _ = _normalize_binding(*hypervectors)
    operands = _broadcast_operands(hvs, is_torch)

    total = operands[0]
    for operand in operands[1:]:
        total = total + operand

    if is_torch:
        return torch.fmod(total, 2 * pi)
    return np.mod(total, 2 * pi)


def ElementAngleSubtraction(*hypervectors: ArrayLike) -> ArrayLike:
    """
    Unbinding by subtracting phase angles.

    Inverse operation for ElementAngleAddition. Operands broadcast over the
    trailing batch axes.

    Args:
        *hypervectors: Variable number of phase hypervectors, or single 2D batch

    Returns:
        Unbound hypervector with phases in [0, 2*pi)
    """
    hvs, is_torch, _ = _normalize_binding(*hypervectors)
    operands = _broadcast_operands(hvs, is_torch)

    diff = operands[0]
    for operand in operands[1:]:
        diff = diff - operand

    if is_torch:
        return torch.fmod(diff, 2 * pi)
    return np.mod(diff, 2 * pi)
