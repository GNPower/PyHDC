from math import pi

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
# Angle-based Binding (for FHRR)
# ============================================================================


def ElementAngleAddition(*hypervectors: ArrayLike) -> ArrayLike:
    """
    Binding by adding phase angles.

    Used with Fourier Holographic Reduced Representations (FHRR) where
    hypervector elements represent phase angles.

    Args:
        *hypervectors: Variable number of phase hypervectors, or single 2D batch

    Returns:
        Bound hypervector with phases in [0, 2Ï€)
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)

    if is_torch:
        total = torch.sum(torch.stack(hvs), dim=0)
        return torch.fmod(total, 2 * pi)
    else:
        total = np.add.reduce(hvs)
        return np.mod(total, 2 * pi)


def ElementAngleSubtraction(*hypervectors: ArrayLike) -> ArrayLike:
    """
    Unbinding by subtracting phase angles.

    Inverse operation for ElementAngleAddition.

    Args:
        *hypervectors: Variable number of phase hypervectors, or single 2D batch

    Returns:
        Unbound hypervector with phases in [0, 2Ï€)
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)

    if is_torch:
        diff = torch.subtract(hvs[0], torch.sum(torch.stack(hvs[1:]), dim=0))
        return torch.fmod(diff, 2 * pi)
    else:
        diff = np.subtract.reduce(hvs)
        return np.mod(diff, 2 * pi)
