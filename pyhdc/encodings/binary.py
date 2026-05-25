#!/usr/bin/env python
"""
Binary and Sparse Binary Encodings for HDC

HDC-compatible wrapper for BSC and BSDC encodings.
"""

from functools import partial
from typing import Any, Callable, Optional

import numpy as np

from pyhdc.components.binding import (
    AdditiveContextDependentThinning,
    ExclusiveOr,
    InverseSegmentShifting,
    InverseShifting,
    SegmentShifting,
    Shifting,
)
from pyhdc.components.bundling import (
    Disjunction,
    DisjunctionThinned,
    ElementAdditionBinaryThreshold,
)
from pyhdc.components.elements import BernoulliBinary, BernoulliSparse, SparseSegmented
from pyhdc.components.similarity import HammingDistance, Overlap
from pyhdc.components.thinning import NoThin
from pyhdc.encodings.base import Encoding
from pyhdc.exceptions import RaiseNotImplementedError
from pyhdc.generation.base import HDCGenerator
from pyhdc.hypervector import EncodingSpec
from pyhdc.types import Backend, Device

# ============================================================================
# Binary and Sparse Binary Encodings
# ============================================================================


class BSC(Encoding):
    """
    Binary Spatter Code.

    Uses binary values with XOR binding and Hamming distance similarity.
    """

    def __init__(
        self,
        dimension: int = 10_000,
        backend: Backend = "numpy",
        device: Optional[Device] = None,
        dtype: Optional[Any] = None,
        mask: Optional[int] = None,
        generator: Optional[HDCGenerator] = None,
        similarity_remap: Optional[Callable] = None,
        random_choice_range: Optional[float] = None,
    ) -> None:
        self._random_choice_range = random_choice_range
        super().__init__(
            dimension, backend, device, dtype, mask, generator, similarity_remap
        )

    def _get_encoding_spec(self) -> EncodingSpec:
        # Use functools.partial to bake in random_choice_range parameter
        if self._random_choice_range is not None:
            bundling_fn = partial(
                ElementAdditionBinaryThreshold,
                random_choice_range=self._random_choice_range,
            )
        else:
            bundling_fn = ElementAdditionBinaryThreshold

        return EncodingSpec(
            dtype=np.int8,
            element_generator=BernoulliBinary,
            similarity_fn=HammingDistance,
            bundling_fn=bundling_fn,
            thinning_fn=NoThin,
            binding_fn=ExclusiveOr,
            unbinding_fn=ExclusiveOr,
            generator_output_type="bits",
        )


class BSDC_CDT(Encoding):
    """
    Binary Sparse Distributed Code with Context-Dependent Thinning.

    Uses sparse binary representations with context-dependent thinning.
    """

    def _get_encoding_spec(self) -> EncodingSpec:
        return EncodingSpec(
            dtype=np.int8,
            element_generator=BernoulliSparse,
            similarity_fn=Overlap,
            bundling_fn=Disjunction,
            thinning_fn=NoThin,
            binding_fn=AdditiveContextDependentThinning,
            unbinding_fn=RaiseNotImplementedError,
            generator_output_type="bits",
        )


class BSDC_S(Encoding):
    """
    Binary Sparse Distributed Code with Shifting.

    Uses sparse binary representations with circular shifting for binding.
    """

    def _get_encoding_spec(self) -> EncodingSpec:
        return EncodingSpec(
            dtype=np.int8,
            element_generator=BernoulliSparse,
            similarity_fn=Overlap,
            bundling_fn=Disjunction,
            thinning_fn=NoThin,
            binding_fn=Shifting,
            unbinding_fn=InverseShifting,
            generator_output_type="bits",
        )


class BSDC_SEG(Encoding):
    """
    Binary Sparse Distributed Code with Segment Shifting.

    Uses sparse segmented representations with segment-wise shifting.
    """

    def _get_encoding_spec(self) -> EncodingSpec:
        return EncodingSpec(
            dtype=np.int8,
            element_generator=SparseSegmented,
            similarity_fn=Overlap,
            bundling_fn=Disjunction,
            thinning_fn=NoThin,
            binding_fn=SegmentShifting,
            unbinding_fn=InverseSegmentShifting,
            generator_output_type="bits",
        )


class BSDC_THIN(Encoding):
    """
    Binary Sparse Distributed Code with post-bundling thinning (BSDC-THIN).

    After bundling via bitwise OR, randomly zeros bits to keep the fraction
    of 1-bits at most `density`. This controls density growth from repeated
    bundling.

    Args:
        density: Maximum output density after bundling, defaults to 0.5
    """

    def __init__(
        self,
        dimension: int = 10_000,
        backend: Backend = "numpy",
        device: Optional[Device] = None,
        dtype: Optional[Any] = None,
        mask: Optional[int] = None,
        generator: Optional[HDCGenerator] = None,
        similarity_remap: Optional[Callable] = None,
        density: float = 0.5,
    ) -> None:
        self._density = density
        super().__init__(
            dimension, backend, device, dtype, mask, generator, similarity_remap
        )

    def _get_encoding_spec(self) -> EncodingSpec:
        bundling_fn = partial(DisjunctionThinned, density=self._density)
        return EncodingSpec(
            dtype=np.int8,
            element_generator=BernoulliSparse,
            similarity_fn=Overlap,
            bundling_fn=bundling_fn,
            thinning_fn=NoThin,
            binding_fn=Shifting,
            unbinding_fn=InverseShifting,
            generator_output_type="bits",
        )
