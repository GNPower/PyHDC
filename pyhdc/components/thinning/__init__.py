#!/usr/bin/env python
"""Binding operations for hypervectors."""

import numpy as np


def NoThin(hypervector: np.ndarray) -> np.ndarray:
    return hypervector


# Import from submodules
from pyhdc.components.thinning.random import Random, SegmentedRandom
from pyhdc.components.thinning.sumset import SegmentedSumset, Sumset

# Define what gets exported with "from pyhdc.components.thinning import *"
__all__ = [
    "NoThin",
    "Random",
    "SegmentedRandom",
    "Sumset",
    "SegmentedSumset",
]
