#!/usr/bin/env python
"""
Matrix-Based Encodings for HDC

HDC-compatible wrapper for VTB and MBAT encodings.
"""

import numpy as np

from pyhdc.components.binding import (
    InverseMatrixMultiplication,
    MatrixMultiplication,
    TransposeVectorDerivedTransformation,
    VectorDerivedTransformation,
)
from pyhdc.components.bundling import ElementAdditionNormalized
from pyhdc.components.elements import NormalReal
from pyhdc.components.similarity import CosineSimilarity
from pyhdc.components.thinning import NoThin
from pyhdc.components.unary import L2Normalize, Negate
from pyhdc.encodings.base import Encoding
from pyhdc.hypervector import EncodingSpec

# ============================================================================
# Matrix-Based Encodings
# ============================================================================


class VTB(Encoding):
    """
    Vector-derived Transformation Binding.

    Uses vector-derived transformations for binding operations.
    """

    def _get_encoding_spec(self) -> EncodingSpec:
        return EncodingSpec(
            dtype=np.float32,
            element_generator=NormalReal,
            similarity_fn=CosineSimilarity,
            bundling_fn=ElementAdditionNormalized,
            thinning_fn=NoThin,
            binding_fn=VectorDerivedTransformation,
            unbinding_fn=TransposeVectorDerivedTransformation,
            generator_output_type="floats",
            normalize_fn=L2Normalize,
            negative_fn=Negate,
        )


class MBAT(Encoding):
    """
    Matrix Binding of Additive Terms.

    Uses matrix multiplication for binding operations.
    """

    def _get_encoding_spec(self) -> EncodingSpec:
        return EncodingSpec(
            dtype=np.float32,
            element_generator=NormalReal,
            similarity_fn=CosineSimilarity,
            bundling_fn=ElementAdditionNormalized,
            thinning_fn=NoThin,
            binding_fn=MatrixMultiplication,
            unbinding_fn=InverseMatrixMultiplication,
            generator_output_type="floats",
            normalize_fn=L2Normalize,
            negative_fn=Negate,
        )
