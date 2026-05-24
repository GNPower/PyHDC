#!/usr/bin/env python
"""
Holographic Encodings for HDC

HDC-compatible wrapper for HRR and FHRR encodings.
"""

from functools import partial
from typing import Any, Optional

import numpy as np

from pyhdc.components.binding import (
    CircularConvolution,
    CircularCorrelation,
    ElementAngleAddition,
    ElementAngleSubtraction,
)
from pyhdc.components.bundling import (
    AnglesOfElementAddition,
    ElementAddition,
    ElementAdditionConstantNormalized,
    ElementAdditionNormalized,
)
from pyhdc.components.elements import NormalReal, UniformAngles
from pyhdc.components.similarity import AngleDistance, CosineSimilarity
from pyhdc.components.thinning import NoThin
from pyhdc.encodings.base import Encoding
from pyhdc.generation.base import HDCGenerator
from pyhdc.hypervector import EncodingSpec
from pyhdc.types import Backend, Device

# ============================================================================
# Holographic Encodings
# ============================================================================


class HRR(Encoding):
    """
    Holographic Reduced Representation.

    Uses circular convolution for binding and normalized bundling.

    Args:
        random_choice_range: Optional float (rho). When set, coordinates whose
            |pre-norm sum| <= rho * sqrt(N) are replaced by independent N(0,1)
            draws before normalization (band randomization).
    """

    def __init__(
        self,
        dimension: int = 10_000,
        backend: Backend = "numpy",
        device: Optional[Device] = None,
        dtype: Optional[Any] = None,
        mask: Optional[int] = None,
        generator: Optional[HDCGenerator] = None,
        random_choice_range: Optional[float] = None,
    ) -> None:
        self._random_choice_range = random_choice_range
        super().__init__(dimension, backend, device, dtype, mask, generator)

    def _get_encoding_spec(self) -> EncodingSpec:
        if self._random_choice_range is not None:
            bundling_fn = partial(
                ElementAdditionNormalized, random_choice_range=self._random_choice_range
            )
        else:
            bundling_fn = ElementAdditionNormalized
        return EncodingSpec(
            dtype=np.float32,
            element_generator=NormalReal,
            similarity_fn=CosineSimilarity,
            bundling_fn=bundling_fn,
            thinning_fn=NoThin,
            binding_fn=CircularConvolution,
            unbinding_fn=CircularCorrelation,
            generator_output_type="floats",
        )


class HRR_NoNorm(Encoding):
    """HRR without normalized bundling."""

    def _get_encoding_spec(self) -> EncodingSpec:
        return EncodingSpec(
            dtype=np.float32,
            element_generator=NormalReal,
            similarity_fn=CosineSimilarity,
            bundling_fn=ElementAddition,
            thinning_fn=NoThin,
            binding_fn=CircularConvolution,
            unbinding_fn=CircularCorrelation,
            generator_output_type="floats",
        )


class HRR_ConstNorm(Encoding):
    """HRR with constant normalization for bundling."""

    def _get_encoding_spec(self) -> EncodingSpec:
        return EncodingSpec(
            dtype=np.float32,
            element_generator=NormalReal,
            similarity_fn=CosineSimilarity,
            bundling_fn=ElementAdditionConstantNormalized,
            thinning_fn=NoThin,
            binding_fn=CircularConvolution,
            unbinding_fn=CircularCorrelation,
            generator_output_type="floats",
        )


class FHRR(Encoding):
    """
    Fourier Holographic Reduced Representation.

    Uses angular/phase representations with angle-based operations.

    Args:
        random_choice_range: Optional float (rho). When set, coordinates whose
            phasor magnitude <= rho * sqrt(N/2) are replaced by independent
            Uniform[-pi, pi] draws (band randomization).
    """

    def __init__(
        self,
        dimension: int = 10_000,
        backend: Backend = "numpy",
        device: Optional[Device] = None,
        dtype: Optional[Any] = None,
        mask: Optional[int] = None,
        generator: Optional[HDCGenerator] = None,
        random_choice_range: Optional[float] = None,
    ) -> None:
        self._random_choice_range = random_choice_range
        super().__init__(dimension, backend, device, dtype, mask, generator)

    def _get_encoding_spec(self) -> EncodingSpec:
        if self._random_choice_range is not None:
            bundling_fn = partial(
                AnglesOfElementAddition, random_choice_range=self._random_choice_range
            )
        else:
            bundling_fn = AnglesOfElementAddition
        return EncodingSpec(
            dtype=np.float32,
            element_generator=UniformAngles,
            similarity_fn=AngleDistance,
            bundling_fn=bundling_fn,
            thinning_fn=NoThin,
            binding_fn=ElementAngleAddition,
            unbinding_fn=ElementAngleSubtraction,
            generator_output_type="floats",
        )
