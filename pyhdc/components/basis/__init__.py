#!/usr/bin/env python
"""Family-aware basis builders for data encoders.

Each builder has the signature ``fn(encoding, count, dimension=None) -> (D, count)``
array in the encoding's value domain and backend. ``dimension`` defaults to
``encoding.dimension``. These are the building blocks the codebook encoders
(Level, Thermometer, Circular, ...) hold as their basis.
"""

from pyhdc.components.basis.codebook import (
    circular,
    empty,
    identity,
    level,
    random,
    thermometer,
)
from pyhdc.components.basis.domain import binding_identity, family_endpoints

__all__ = [
    "empty",
    "identity",
    "random",
    "level",
    "circular",
    "thermometer",
    "family_endpoints",
    "binding_identity",
]
