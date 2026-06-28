#!/usr/bin/env python
"""Multiplicative multiset binding: reduce a stacked set of hypervectors by product.

Operates on raw arrays dimension-first (axis 0 is the dimension ``D``, the trailing
axis is the batch). This is the element-wise binding of a set of bipolar/MAP
hypervectors; for non-multiplicative binders (XOR, convolution, matrix, ...) use
``Encoding.bind`` instead.
"""

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None


def _is_torch(data):
    return TORCH_AVAILABLE and torch is not None and torch.is_tensor(data)


def multibind(data, axis=-1):
    """Multiplicative bind: product over a stacked batch axis (defaults to the last).

    Binds a set of hypervectors element-wise, e.g. ``(D, N) -> (D,)``.
    """
    if _is_torch(data):
        return data.prod(dim=axis)
    return data.prod(axis=axis)
