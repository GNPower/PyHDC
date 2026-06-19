"""
Unary HDC operations: permute, inverse, negative, normalize.

Each operates on a single hypervector (or batch) of raw array data, dimension
first (axis 0 is always the hypervector dimension ``D``), and works on both the
numpy and torch backends.
"""

from math import pi

import numpy as np

# Optional PyTorch support
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.types import ArrayLike


def _is_torch(data: ArrayLike) -> bool:
    return TORCH_AVAILABLE and torch is not None and torch.is_tensor(data)


def CyclicShift(data: ArrayLike, shift: int = 1) -> ArrayLike:
    """
    Cyclic-shift permutation along axis 0 (the dimension); broadcasts over any
    trailing batch axes. Encoding-agnostic, so it is the default ``permute``.
    """
    amount = int(shift)
    if _is_torch(data):
        return torch.roll(data, shifts=amount, dims=0)
    return np.roll(data, amount, axis=0)


def IdentityInverse(data: ArrayLike) -> ArrayLike:
    """
    Inverse for self-inverse binding (MAP bipolar multiply, BSC XOR): the element
    is its own inverse. Exact for bipolar/binary values.
    """
    return data


def ReverseInverse(data: ArrayLike) -> ArrayLike:
    """
    Exact involution inverse of circular convolution (HRR): keep index 0 and
    reverse the remaining coordinates along axis 0.
    """
    if _is_torch(data):
        return torch.cat([data[:1], torch.flip(data[1:], dims=[0])], dim=0)
    return np.concatenate([data[:1], np.flip(data[1:], axis=0)], axis=0)


def PhaseNegate(data: ArrayLike) -> ArrayLike:
    """Inverse for FHRR angle binding: negate the phase (mod 2*pi)."""
    if _is_torch(data):
        return torch.remainder(-data, 2 * pi)
    return np.mod(-data, 2 * pi)


def Negate(data: ArrayLike) -> ArrayLike:
    """Additive (bundling) inverse: element-wise negation."""
    return -data


def L2Normalize(data: ArrayLike) -> ArrayLike:
    """Normalize each hypervector to unit L2 length along axis 0."""
    if _is_torch(data):
        return data / torch.norm(data, dim=0, keepdim=True)
    return data / np.linalg.norm(data, axis=0, keepdims=True)


def WrapPhase(data: ArrayLike) -> ArrayLike:
    """Normalize FHRR phases to the canonical range [-pi, pi)."""
    if _is_torch(data):
        return torch.remainder(data + pi, 2 * pi) - pi
    return np.mod(data + pi, 2 * pi) - pi


def SignNormalize(data: ArrayLike) -> ArrayLike:
    """Normalize MAP hypervectors back to bipolar {-1, 0, +1} by sign."""
    if _is_torch(data):
        return torch.sign(data)
    return np.sign(data)
