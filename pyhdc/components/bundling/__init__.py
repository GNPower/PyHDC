#!/usr/bin/env python
"""Bundling operations for hypervectors."""

# Import from submodules
from pyhdc.components.bundling.addition import (
    ElementAddition,
    ElementAdditionBits,
    ElementAdditionCut,
)
from pyhdc.components.bundling.angles import AnglesOfElementAddition
from pyhdc.components.bundling.binary import Disjunction, DisjunctionThinned
from pyhdc.components.bundling.multiset import multibundle, multiset
from pyhdc.components.bundling.normalized import (
    ElementAdditionConstantNormalized,
    ElementAdditionNormalized,
)
from pyhdc.components.bundling.selection import multirandsel, randsel
from pyhdc.components.bundling.threshold import (
    ElementAdditionBinaryThreshold,
    ElementAdditionBipolarThreshold,
)

# Define what gets exported with "from pyhdc.components.bundling import *"
__all__ = [
    "ElementAddition",
    "ElementAdditionBits",
    "ElementAdditionCut",
    "AnglesOfElementAddition",
    "Disjunction",
    "DisjunctionThinned",
    "ElementAdditionNormalized",
    "ElementAdditionConstantNormalized",
    "ElementAdditionBinaryThreshold",
    "ElementAdditionBipolarThreshold",
    "randsel",
    "multirandsel",
    "multiset",
    "multibundle",
]
