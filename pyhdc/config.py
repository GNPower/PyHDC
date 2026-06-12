#!/usr/bin/env python
"""
Global backend and device preferences for PyHDC.

By default PyHDC uses the NumPy backend on the CPU. These helpers let a script
change that default once, near the top, so encodings created afterwards inherit
the preferred backend/device without passing ``backend=``/``device=`` every time::

    import pyhdc

    pyhdc.prefer_cuda()                  # all new encodings use torch on cuda
    enc = pyhdc.MAP_I(dimension=10_000)  # already on the GPU

Only encodings created after the preference is set are affected; an explicit
``backend``/``device`` argument always overrides the global default.
"""

from typing import Optional

from pyhdc.types import Backend, Device

# Optional PyTorch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


_DEFAULT_BACKEND: Backend = "numpy"
_DEFAULT_DEVICE: Optional[Device] = None


def get_default_backend() -> Backend:
    """Return the current global default backend ('numpy' or 'torch')."""
    return _DEFAULT_BACKEND


def get_default_device() -> Optional[Device]:
    """Return the current global default device (None for the numpy backend)."""
    return _DEFAULT_DEVICE


def prefer_torch(device: Optional[Device] = None) -> None:
    """
    Make ``torch`` the default backend for new encodings.

    Args:
        device: Optional default device (e.g. ``"cpu"`` or ``"cuda"``). Left
            unchanged if None.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "prefer_torch() requires PyTorch. Install it with: pip install torch"
        )
    global _DEFAULT_BACKEND, _DEFAULT_DEVICE
    _DEFAULT_BACKEND = "torch"
    if device is not None:
        _DEFAULT_DEVICE = device


def prefer_cuda(index: Optional[int] = None) -> None:
    """
    Make ``torch`` the default backend on a CUDA device for new encodings.

    Args:
        index: Optional CUDA device index (e.g. ``0`` for ``"cuda:0"``). Uses
            ``"cuda"`` if None.

    Raises:
        ImportError: If PyTorch is not installed.
        RuntimeError: If no CUDA device is available.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "prefer_cuda() requires PyTorch. Install it with: pip install torch"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("prefer_cuda() requires a CUDA-capable device, none found.")
    global _DEFAULT_BACKEND, _DEFAULT_DEVICE
    _DEFAULT_BACKEND = "torch"
    _DEFAULT_DEVICE = f"cuda:{index}" if index is not None else "cuda"


def prefer_numpy() -> None:
    """Reset the default backend to NumPy on the CPU."""
    global _DEFAULT_BACKEND, _DEFAULT_DEVICE
    _DEFAULT_BACKEND = "numpy"
    _DEFAULT_DEVICE = None


def prefer_cpu() -> None:
    """
    Pin the default device to the CPU.

    Affects the torch backend only; the backend itself is left unchanged (use
    ``prefer_numpy()`` to switch back to NumPy).
    """
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = "cpu"
