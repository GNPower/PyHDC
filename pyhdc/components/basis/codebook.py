#!/usr/bin/env python
"""Family-aware basis builders.

Each builder returns a raw ``(D, count)`` array (numpy or torch, matching the
encoding's backend) whose columns are basis hypervectors in the encoding's value
domain. Builders are component-level and return arrays, not Hypervectors.

- ``empty`` / ``random`` / ``identity`` are trivial draws.
- ``level`` / ``circular`` are family-agnostic. They mix two ordinary
  ``encoding.generate`` draws with a per-coordinate threshold, so they work for every
  family without any per-family special-casing.
- ``thermometer`` is a deterministic cumulative code and needs the
  discrete ``(low, high)`` endpoints, so it is defined only for discrete families.
"""

import numpy as np

from pyhdc.components.basis.domain import binding_identity, family_endpoints

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None


def _resolve_dim(encoding, dimension):
    return encoding.dimension if dimension is None else int(dimension)


def random(encoding, count, dimension=None):
    """``count`` independent random hypervectors as a ``(D, count)`` codebook."""
    dim = _resolve_dim(encoding, dimension)
    return encoding.generate((dim, count)).data


def identity(encoding, count, dimension=None):
    """``count`` copies of the binding-identity element as a ``(D, count)`` codebook.

    The binding-identity ``e`` satisfies ``bind(x, e) == x``, such that every
    column is ``e``. Defined for the MAP, HRR, FHRR, and BSC families,
    raises ``NotImplementedError`` for VTB, MBAT, and the BSDC family
    (no neutral binding element).
    """
    dim = _resolve_dim(encoding, dimension)
    elem = binding_identity(encoding, dim)  # (D,)
    if encoding.backend == "torch":
        return elem[:, None].repeat(1, count)
    return np.repeat(elem[:, None], count, axis=1)


def empty(encoding, count, dimension=None):
    """``count`` all-zero hypervectors as a ``(D, count)`` array."""
    dim = _resolve_dim(encoding, dimension)
    return encoding.zeros((dim, count)).data


def level(encoding, count, dimension=None):
    """A linear level codebook: adjacent columns correlated, ends near-orthogonal.

    Built family-agnostically from two random endpoint draws ``base`` and ``alt`` plus
    a per-coordinate uniform threshold ``u``. Column ``i`` keeps ``base`` where
    ``u >= i / (count - 1)`` and ``alt`` elsewhere, so each coordinate flips from
    ``base`` to ``alt`` exactly once (at its own threshold). Similarity therefore
    decays monotonically with ``|i - j|``. Column 0 is ``base`` and the last column is
    ``alt`` (two independent draws, so near-orthogonal).
    """
    dim = _resolve_dim(encoding, dimension)
    base = encoding.generate(dim).data
    alt = encoding.generate(dim).data
    span = max(count - 1, 1)

    if encoding.backend == "torch":
        assert torch is not None
        idx = torch.arange(count, device=base.device, dtype=torch.float32) / span
        u = torch.rand(dim, device=base.device)
        keep = u[:, None] >= idx[None, :]  # (D, count) bool
        return torch.where(keep, base[:, None], alt[:, None]).to(base.dtype)

    idx = np.arange(count) / span
    u = np.random.uniform(0.0, 1.0, dim)
    keep = u[:, None] >= idx[None, :]
    return np.where(keep, base[:, None], alt[:, None]).astype(base.dtype)


def circular(encoding, count, dimension=None):
    """A circular (ring) level codebook: similarity wraps, so level 0 ~ level L-1.

    Like :func:`level`, but each coordinate is assigned a random start phase ``p`` in
    ``[0, count)`` and takes ``base`` over a half-ring arc and ``alt`` over the other
    half. Similarity depends on ring distance ``min(|i - j|, count - |i - j|)``, so the
    first and last columns are adjacent and the diametrically opposite column is the
    similarity minimum (near-orthogonal, around 0).
    """
    dim = _resolve_dim(encoding, dimension)
    base = encoding.generate(dim).data
    alt = encoding.generate(dim).data
    half = count / 2.0

    if encoding.backend == "torch":
        assert torch is not None
        i = torch.arange(count, device=base.device, dtype=torch.float32)
        p = torch.rand(dim, device=base.device) * count
        d = torch.remainder(i[None, :] - p[:, None], count)  # (D, count) ring offset
        keep = d < half
        return torch.where(keep, base[:, None], alt[:, None]).to(base.dtype)

    i = np.arange(count)
    p = np.random.uniform(0.0, count, dim)
    d = np.mod(i[None, :] - p[:, None], count)
    keep = d < half
    return np.where(keep, base[:, None], alt[:, None]).astype(base.dtype)


def thermometer(encoding, count, dimension=None):
    """A deterministic thermometer (cumulative unary) codebook. Discrete families only.

    Column ``i`` sets its first ``round(i / (count - 1) * D)`` coordinates to the high
    endpoint and the rest to the low endpoint, so each column's high-set is a strict
    superset of the previous column's (nested). Column 0 is the constant all-low vector
    and the last column the constant all-high vector (so the two ends are
    anti-correlated, not orthogonal). Distinct from :func:`level`, whose endpoints are
    two independent random draws.

    Raises:
        NotImplementedError: For continuous and phase families (via
            :func:`~pyhdc.components.basis.domain.family_endpoints`).
    """
    dim = _resolve_dim(encoding, dimension)
    low, high = family_endpoints(encoding)
    span = max(count - 1, 1)

    if encoding.backend == "torch":
        assert torch is not None
        ref = encoding.generate(1).data  # (1,) tensor: correct device + dtype
        device = ref.device
        coord = torch.arange(dim, device=device).unsqueeze(1)  # (D, 1)
        fill = torch.arange(count, device=device, dtype=torch.float32) / span
        cutoff = torch.round(fill * dim).long()  # (count,)
        filled = coord < cutoff.unsqueeze(0)  # (D, count) bool
        return (filled.to(ref.dtype) * (high - low) + low).to(ref.dtype)

    cutoff = np.rint((np.arange(count) / span) * dim).astype(np.int64)
    coord = np.arange(dim)[:, None]
    filled = coord < cutoff[None, :]
    return np.where(filled, high, low).astype(encoding._spec.dtype)
