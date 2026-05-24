#!/usr/bin/env python
"""Similarity operations for hypervectors."""

# Import from submodules
from pyhdc.components.similarity.angle import AngleDistance
from pyhdc.components.similarity.cosine import CosineSimilarity
from pyhdc.components.similarity.hamming import HammingDistance
from pyhdc.components.similarity.overlap import Overlap

# Define what gets exported with "from pyhdc.components.similarity import *"
__all__ = [
    "AngleDistance",
    "CosineSimilarity",
    "HammingDistance",
    "Overlap",
]
