#!/usr/bin/env python
"""Codebook encoders: a value indexes into a precomputed ``(D, L)`` basis.

All six hold a basis Hypervector built by a :mod:`pyhdc.components.basis` builder and
differ only in which builder runs at construction. ``encode`` maps a value to a level
index (clamp + quantize to nearest) and selects that column. A batch of values
encodes to a ``(D, B)`` hypervector. ``Circular`` overrides the index mapping to wrap
modulo ``levels`` instead of clamping.
"""

import numpy as np

from pyhdc.components import basis
from pyhdc.encoders import base


class _CodebookEncoder(base.Encoder):
    """Shared machinery for the codebook encoders."""

    _builder = None  # set by each subclass to a pyhdc.components.basis builder

    def __init__(self, encoding, levels, low=0.0, high=1.0):
        if int(levels) < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        if high <= low:
            raise ValueError(f"high must be > low, got low={low}, high={high}")
        self.levels = int(levels)
        self.low = float(low)
        self.high = float(high)
        super().__init__(encoding)

    def _build_params(self):
        # _builder is set to a basis function on each concrete subclass.
        return type(self)._builder(
            self.encoding, self.levels
        )  # pylint: disable=not-callable

    def _indices(self, value):
        arr, was_scalar = self._value_batch(value)
        return base.value_to_index(arr, self.low, self.high, self.levels), was_scalar

    def encode(self, value):
        idx, was_scalar = self._indices(value)
        cols = self._select_columns(self._params, idx)  # (D, B)
        return self._wrap(cols[:, 0] if was_scalar else cols)


class Empty(_CodebookEncoder):
    """All-zero codebook, every value encodes to a zero hypervector."""

    _builder = staticmethod(basis.empty)


class Identity(_CodebookEncoder):
    """Codebook of the binding-identity element, every value encodes to ``e``.

    The binding-identity ``e`` satisfies ``bind(x, e) == x``. Defined for the MAP,
    HRR, FHRR, and BSC families, raises ``NotImplementedError`` at construction for
    VTB, MBAT, and the BSDC family (no neutral binding element).
    """

    _builder = staticmethod(basis.identity)


class Random(_CodebookEncoder):
    """Codebook of distinct random atoms."""

    _builder = staticmethod(basis.random)


class Level(_CodebookEncoder):
    """Linear level encoder: nearby values map to correlated hypervectors."""

    _builder = staticmethod(basis.level)


class Thermometer(_CodebookEncoder):
    """Cumulative (thermometer) encoder. Discrete (bipolar/binary) families only."""

    _builder = staticmethod(basis.thermometer)


class Circular(_CodebookEncoder):
    """Circular level encoder for periodic values; the index wraps modulo ``levels``."""

    _builder = staticmethod(basis.circular)

    def _indices(self, value):
        arr, was_scalar = self._value_batch(value)
        pos = (np.asarray(arr, dtype=float) - self.low) / (self.high - self.low)
        idx = np.rint(pos * self.levels).astype(np.intp) % self.levels
        return idx, was_scalar
