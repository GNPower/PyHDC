#!/usr/bin/env python
"""Data encoders: map raw data into hypervectors.

An encoder wraps an :class:`~pyhdc.Encoding` instance and turns data (a scalar, a
feature vector, or a batch of either) into a :class:`~pyhdc.Hypervector`. Codebook
encoders index a precomputed basis, functional encoders transform a feature vector.

    >>> import pyhdc
    >>> enc = pyhdc.Level(pyhdc.MAP_I(dimension=10_000), levels=100, low=0.0, high=1.0)
    >>> hv = enc.encode(0.5)            # (D,) Hypervector
    >>> batch = enc.encode([0.1, 0.5])  # (D, 2) Hypervector
"""

from pyhdc.encoders.base import Encoder
from pyhdc.encoders.codebook import (
    Circular,
    Empty,
    Identity,
    Level,
    Random,
    Thermometer,
)
from pyhdc.encoders.functional import Density, FractionalPower, Projection, Sinusoid

__all__ = [
    "Encoder",
    "Empty",
    "Identity",
    "Random",
    "Level",
    "Thermometer",
    "Circular",
    "Projection",
    "Sinusoid",
    "Density",
    "FractionalPower",
]
