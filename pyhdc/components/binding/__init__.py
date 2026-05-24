#!/usr/bin/env python
"""Binding operations for hypervectors."""

# Import from submodules
from pyhdc.components.binding.angles import (
    ElementAngleAddition,
    ElementAngleSubtraction,
)
from pyhdc.components.binding.cdt import AdditiveContextDependentThinning
from pyhdc.components.binding.convolution import (
    CircularConvolution,
    CircularCorrelation,
)
from pyhdc.components.binding.multiplication import (
    ElementMultiplication,
    InverseMatrixMultiplication,
    MatrixMultiplication,
)
from pyhdc.components.binding.shifting import (
    InverseSegmentShifting,
    InverseShifting,
    SegmentShifting,
    Shifting,
)
from pyhdc.components.binding.vtb import (
    TransposeVectorDerivedTransformation,
    VectorDerivedTransformation,
)
from pyhdc.components.binding.xor import ExclusiveOr

# Define what gets exported with "from pyhdc.components.binding import *"
__all__ = [
    "ElementAngleAddition",
    "ElementAngleSubtraction",
    "AdditiveContextDependentThinning",
    "CircularConvolution",
    "CircularCorrelation",
    "ElementMultiplication",
    "MatrixMultiplication",
    "InverseMatrixMultiplication",
    "Shifting",
    "InverseShifting",
    "SegmentShifting",
    "InverseSegmentShifting",
    "VectorDerivedTransformation",
    "TransposeVectorDerivedTransformation",
    "ExclusiveOr",
]
