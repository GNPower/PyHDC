#!/usr/bin/env python
"""HDC Components - building blocks for hypervector operations."""

# Re-export all submodules
from pyhdc.components import binding, bundling, elements, similarity, thinning

# Define what gets exported with "from pyhdc.components import *"
__all__ = [
    "binding",
    "bundling",
    "similarity",
    "elements",
    "thinning",
]
