#!/usr/bin/env python
"""The :class:`Encoder` base class and value/index helpers.

An encoder wraps an :class:`~pyhdc.Encoding` instance and maps data (a scalar, a
feature vector, or a batch of either) to a :class:`~pyhdc.Hypervector` tagged with
that encoding. It reuses the encoding's ``generate``/``normalize_fn``/backend, so
encoder output works with the existing bundle/bind/similarity/select functions.
Everything is dimension-first: a batch of ``B`` inputs encodes to a ``(D, B)``
hypervector.
"""

from abc import ABC, abstractmethod

import numpy as np

from pyhdc.hypervector import Hypervector

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None


def map_range(value, low, high, levels):
    """Affine-map a value (or array) in ``[low, high]`` to a ``[0, levels - 1]`` index.

    Returns a continuous (unrounded, unclamped) position.
    """
    span = max(levels - 1, 1)
    return (np.asarray(value, dtype=float) - low) / (high - low) * span


def value_to_index(value, low, high, levels):
    """Map a value to the nearest level index, clamped to ``[0, levels - 1]``."""
    pos = map_range(value, low, high, levels)
    idx = np.rint(pos).astype(np.intp)
    return np.clip(idx, 0, levels - 1)


def index_to_value(index, low, high, levels):
    """Inverse of :func:`value_to_index`: the representative value of a level index.

    Round-trip is only meaningful for ``levels >= 2``, for ``levels == 1`` this
    returns ``low`` for every index.
    """
    span = max(levels - 1, 1)
    return low + (np.asarray(index, dtype=float) / span) * (high - low)


class Encoder(ABC):
    """Base class for all data encoders.

    Subclasses implement :meth:`_build_params` (build the basis / weight array at
    construction) and :meth:`encode` (map a value or batch to a Hypervector).
    """

    def __init__(self, encoding):
        self.encoding = encoding
        self.dimension = encoding.dimension
        self.backend = encoding.backend
        self.device = encoding.device
        self._params = self._build_params()

    @abstractmethod
    def _build_params(self):
        """Build and return the encoder's parameter array (basis or weights)."""

    @abstractmethod
    def encode(self, value):
        """Encode a value (or batch of values) into a Hypervector."""

    def __call__(self, value):
        return self.encode(value)

    @property
    def params(self):
        """The encoder's parameter array (basis codebook or projection weights)."""
        return self._params

    def _wrap(self, data, metadata=None):
        return Hypervector(
            data,
            self.encoding,
            self.backend,
            metadata if metadata is not None else {"encoder": type(self).__name__},
        )

    def _value_batch(self, value):
        """Coerce a scalar or 1D sequence of values to a ``(B,)`` numpy array.

        Returns ``(values, was_scalar)``; ``was_scalar`` lets ``encode`` squeeze a
        ``(D, 1)`` result back to ``(D,)``.
        """
        if TORCH_AVAILABLE and torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value)
        was_scalar = arr.ndim == 0
        return (arr.reshape(1) if was_scalar else arr.ravel()), was_scalar

    def _feature_batch(self, value):
        """Coerce a feature vector ``(F,)`` or matrix ``(F, B)`` to a backend ``(F, B)``.

        Returns ``(features, was_single)``, a 1D input is one sample.
        """
        if self.backend == "torch":
            assert torch is not None
            x = (
                value
                if torch.is_tensor(value)
                else torch.as_tensor(np.asarray(value, dtype=np.float32))
            )
            x = x.to(device=self.device).float()
            if x.ndim == 0:
                x = x.reshape(1)
            was_single = x.ndim == 1
            return (x[:, None] if was_single else x), was_single
        x = np.asarray(value, dtype=np.float32)
        if x.ndim == 0:
            x = x.reshape(1)
        was_single = x.ndim == 1
        return (x[:, None] if was_single else x), was_single

    def _select_columns(self, params, idx):
        """Select columns ``idx`` (a ``(B,)`` index array) from a ``(D, L)`` array."""
        if self.backend == "torch":
            assert torch is not None
            sel = torch.as_tensor(np.asarray(idx)).to(
                device=params.device, dtype=torch.long
            )
            return params.index_select(1, sel)
        return params[:, np.asarray(idx, dtype=np.intp)]
