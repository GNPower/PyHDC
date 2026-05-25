#!/usr/bin/env python
"""Similarity operations for hypervectors."""

# Import from submodules
from pyhdc.components.similarity.angle import AngleDistance
from pyhdc.components.similarity.cosine import CosineSimilarity
from pyhdc.components.similarity.hamming import HammingDistance
from pyhdc.components.similarity.overlap import Overlap
from pyhdc.types import ArrayLike


def remap_to_unit(sim: ArrayLike) -> ArrayLike:
    """Remap similarity from [-1, 1] to [0, 1].

    Maps the standard [-1, 1] similarity range to [0, 1], where 0.5 is
    orthogonal, 1.0 is identical, and 0.0 is completely opposite. Works
    on scalars, numpy arrays, and torch tensors.

    Pass to an encoding via the ``similarity_remap`` parameter:

        enc = BSC(dimension=10_000, similarity_remap=remap_to_unit)

    Args:
        sim: Similarity value(s) in [-1, 1]

    Returns:
        Remapped value(s) in [0, 1]
    """
    return (sim + 1) / 2


# Define what gets exported with "from pyhdc.components.similarity import *"
__all__ = [
    "AngleDistance",
    "CosineSimilarity",
    "HammingDistance",
    "Overlap",
    "remap_to_unit",
]
