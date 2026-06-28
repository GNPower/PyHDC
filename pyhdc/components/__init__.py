#!/usr/bin/env python
"""HDC Components - building blocks for hypervector operations."""

# Re-export all submodules
from pyhdc.components import (
    basis,
    binding,
    bundling,
    elements,
    ridealongs,
    similarity,
    thinning,
)
from pyhdc.components.ridealongs import (
    hard_quantize,
    multibind,
    multibundle,
    multirandsel,
    multiset,
    randsel,
    soft_quantize,
)

# Define what gets exported with "from pyhdc.components import *"
__all__ = [
    "basis",
    "binding",
    "bundling",
    "similarity",
    "elements",
    "ridealongs",
    "thinning",
    # C4-C6 ride-along callables
    "randsel",
    "multirandsel",
    "multiset",
    "multibundle",
    "multibind",
    "hard_quantize",
    "soft_quantize",
]
