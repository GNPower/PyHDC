#!/usr/bin/env python
"""Random-selection bundling.

Bundles a set of hypervectors by copying each coordinate from one randomly chosen
input vector, rather than summing them. Operates on raw arrays dimension-first (axis 0
is the dimension ``D``, the trailing axis is the batch).
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
        data: A ``(D, N)`` array (numpy or torch) whose columns are the inputs to
            select among.
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
