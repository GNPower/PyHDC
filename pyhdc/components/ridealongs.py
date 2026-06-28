#!/usr/bin/env python
"""Small composable hypervector helpers.

These operate on raw arrays dimension-first (axis 0 is the dimension ``D``, the
trailing axis is the batch), matching the rest of ``pyhdc.components``. They are
independent of the encoder object model.

- ``randsel`` / ``multirandsel``: random-selection bundling (pick each coordinate
  from a uniformly random input vector).
- ``multiset`` / ``multibundle`` / ``multibind``: reduce a stacked batch axis by
  summation (additive multiset) or product (multiplicative bind).
- ``hard_quantize`` / ``soft_quantize``: map to bipolar ``{-1, +1}`` by sign, or a
  smooth ``tanh`` surrogate.
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


def randsel(data, p=None):
    """Random-selection bundling of a ``(D, N)`` batch into a single ``(D,)`` vector.

    Each coordinate is copied from one of the ``N`` input columns, chosen
    independently at random (uniformly, or per the probability weights ``p`` over the
    ``N`` columns).

    Args:
        data: A ``(D, N)`` array (numpy or torch) where columns are the inputs to select
            among.
        p: Optional length-``N`` weights over the columns (default uniform). Weights
            are normalized to a probability distribution, so they need not sum to 1.

    Returns:
        A ``(D,)`` array of the same backend/dtype as ``data``.
    """
    dim, num = data.shape
    if p is not None:
        weights = np.asarray(p, dtype=np.float64)
        p = weights / weights.sum()  # normalize so both backends agree
    if _is_torch(data):
        rows = torch.arange(dim, device=data.device)
        if p is None:
            idx = torch.randint(0, num, (dim,), device=data.device)
        else:
            tw = torch.as_tensor(p).to(data.device)
            idx = torch.multinomial(tw.expand(dim, num), 1).squeeze(1)
        return data[rows, idx]
    idx = np.random.choice(num, size=dim, p=p)
    return data[np.arange(dim), idx]


def multirandsel(data, count, p=None):
    """Produce ``count`` independent :func:`randsel` draws as a ``(D, count)`` array."""
    cols = [randsel(data, p=p) for _ in range(count)]
    if _is_torch(data):
        return torch.stack(cols, dim=1)
    return np.stack(cols, axis=1)


def multiset(data, axis=-1):
    """Additive multiset: sum a stacked batch axis (defaults to the last axis).

    The additive bundling of a set of hypervectors. For family-specific bundling
    (thresholding, normalization, thinning) use ``Encoding.bundle`` instead.
    """
    if _is_torch(data):
        return data.sum(dim=axis)
    return data.sum(axis=axis)


# Alias: multibundle is the additive multiset.
multibundle = multiset


def multibind(data, axis=-1):
    """Multiplicative bind: product over a stacked batch axis (defaults to the last).

    The element-wise binding of a set of bipolar/MAP hypervectors. For non-
    multiplicative binders (XOR, convolution, matrix, ...) use ``Encoding.bind``.
    """
    if _is_torch(data):
        return data.prod(dim=axis)
    return data.prod(axis=axis)


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
