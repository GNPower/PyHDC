#!/usr/bin/env python
"""Additive multiset bundling: reduce a stacked set of hypervectors by summation.

Operates on raw arrays dimension-first (axis 0 is the dimension ``D``, the trailing
axis is the batch). For family-specific bundling (thresholding, normalization,
thinning) use ``Encoding.bundle`` instead.
"""

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None


def _is_torch(data):
    return TORCH_AVAILABLE and torch is not None and torch.is_tensor(data)


def multiset(data, axis=-1):
    """Additive multiset: sum a stacked batch axis (defaults to the last axis).

    The additive bundling of a set of hypervectors, e.g. ``(D, N) -> (D,)``.
    """
    if _is_torch(data):
        return data.sum(dim=axis)
    return data.sum(axis=axis)


# multibundle is the additive multiset under its conventional HDC name.
multibundle = multiset
