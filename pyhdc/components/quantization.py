#!/usr/bin/env python
"""Quantization of hypervector elements.

Maps continuous element values to a bipolar form: a hard sign quantizer, or a smooth
``tanh`` surrogate. Operates on raw arrays dimension-first, on both backends.
"""

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None


def _is_torch(data):
    return TORCH_AVAILABLE and torch is not None and torch.is_tensor(data)


def hard_quantize(data):
    """Quantize to ``{-1, 0, +1}`` by sign."""
    if _is_torch(data):
        return torch.sign(data)
    return np.sign(data)


def soft_quantize(data, temperature=1.0):
    """Smooth bipolar surrogate: ``tanh(data / temperature)`` in ``(-1, 1)``."""
    if _is_torch(data):
        return torch.tanh(data / temperature)
    return np.tanh(data / temperature)
