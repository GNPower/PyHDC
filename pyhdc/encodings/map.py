#!/usr/bin/env python
"""
Multiply-Add-Permute Encodings for HDC

HDC-compatible wrapper for MAP encodings.
"""

from functools import partial
from typing import Any, Callable, Optional

import numpy as np

from pyhdc.components.binding import ElementMultiplication
from pyhdc.components.bundling import (
    ElementAddition,
    ElementAdditionBipolarThreshold,
    ElementAdditionBits,
    ElementAdditionCut,
)
from pyhdc.components.elements import BernoulliBiploar, UniformBipolar
from pyhdc.components.similarity import CosineSimilarity
from pyhdc.components.thinning import NoThin
from pyhdc.encodings.base import Encoding
from pyhdc.generation.base import HDCGenerator
from pyhdc.hypervector import EncodingSpec
from pyhdc.types import Backend, Device

# ============================================================================
# Multiply-Add-Permute Encodings
# ============================================================================


class MAP_C(Encoding):
    """
    Multiply-Add-Permute encoding with continuous values.

    Uses bipolar values with element-wise multiplication for binding
    and cosine similarity for comparison.

    Args:
        random_choice_range: Optional float (rho). When set, coordinates whose
            |pre-aggregate sum| <= rho * sqrt(N/3) are replaced by independent
            Uniform[-1,1] draws during bundling (band randomization).
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
        if self._random_choice_range is not None:
            bundling_fn = partial(
                ElementAdditionCut, random_choice_range=self._random_choice_range
            )
        else:
            bundling_fn = ElementAdditionCut
        return EncodingSpec(
            dtype=np.float32,
            element_generator=UniformBipolar,
            similarity_fn=CosineSimilarity,
            bundling_fn=bundling_fn,
            thinning_fn=NoThin,
            binding_fn=ElementMultiplication,
            unbinding_fn=ElementMultiplication,
            generator_output_type="floats",
        )


class MAP_I(Encoding):
    """
    Multiply-Add-Permute encoding with integer values.

    Uses bipolar integer values with element-wise multiplication for binding
    and cosine similarity for comparison.

    Args:
        random_choice_range: Optional float (rho). When set, coordinates whose
            |pre-aggregate sum| <= rho * sqrt(N) are replaced by independent
            {-1, +1} draws during bundling (band randomization).
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
        if self._random_choice_range is not None:
            bundling_fn = partial(
                ElementAddition, random_choice_range=self._random_choice_range
            )
        else:
            bundling_fn = ElementAddition
        return EncodingSpec(
            dtype=np.int32,
            element_generator=BernoulliBiploar,
            similarity_fn=CosineSimilarity,
            bundling_fn=bundling_fn,
            thinning_fn=NoThin,
            binding_fn=ElementMultiplication,
            unbinding_fn=ElementMultiplication,
            generator_output_type="bits",
        )


class MAP_I_Bits(Encoding):
    """
    Multiply-Add-Permute encoding with bit-limited integer values.

    Similar to MAP_I but with configurable bit limits via mask parameter.
    """

    def __init__(
        self,
        dimension: int = 10_000,
        backend: Backend = "numpy",
        device: Optional[Device] = None,
        dtype: Optional[Any] = None,
        mask: int = (2**32) - 1,
        generator: Optional[HDCGenerator] = None,
        similarity_remap: Optional[Callable] = None,
    ) -> None:
        self._mask = mask
        super().__init__(
            dimension, backend, device, dtype, mask, generator, similarity_remap
        )

    def _get_encoding_spec(self) -> EncodingSpec:
        bundling_fn = partial(
            ElementAdditionBits,
            min_val=np.iinfo(np.int32).min,
            max_val=np.iinfo(np.int32).max,
        )
        return EncodingSpec(
            dtype=np.int32,
            element_generator=BernoulliBiploar,
            similarity_fn=CosineSimilarity,
            bundling_fn=bundling_fn,
            thinning_fn=NoThin,
            binding_fn=ElementMultiplication,
            unbinding_fn=ElementMultiplication,
            mask=self._mask,
            generator_output_type="words",
        )


class MAP_B(Encoding):
    """
    Multiply-Add-Permute with bipolar thresholding.

    Uses bipolar values with thresholding during bundling.

    Args:
        random_choice_range: Optional float (rho). When set, coordinates whose
            |bipolar sum| <= rho * sqrt(N) are replaced by independent {-1, +1}
            draws during bundling (band randomization).
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
        if self._random_choice_range is not None:
            bundling_fn = partial(
                ElementAdditionBipolarThreshold,
                random_choice_range=self._random_choice_range,
            )
        else:
            bundling_fn = ElementAdditionBipolarThreshold
        return EncodingSpec(
            dtype=np.int8,
            element_generator=BernoulliBiploar,
            similarity_fn=CosineSimilarity,
            bundling_fn=bundling_fn,
            thinning_fn=NoThin,
            binding_fn=ElementMultiplication,
            unbinding_fn=ElementMultiplication,
            generator_output_type="bits",
        )
