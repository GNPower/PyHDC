#!/usr/bin/env python
"""HDC Components - building blocks for hypervector operations."""

# Re-export all submodules
from pyhdc.components import (
    basis,
    binding,
    bundling,
    elements,
    quantization,
    similarity,
    thinning,
)

# Define what gets exported with "from pyhdc.components import *"
__all__ = [
    "basis",
    "binding",
    "bundling",
    "similarity",
    "elements",
    "quantization",
    "thinning",
]
