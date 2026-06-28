#!/usr/bin/env python
"""
Multiply-Add-Permute Encodings for HDC

HDC-compatible wrapper for MAP encodings.
"""

from functools import partial
from typing import Any, Callable, Optional, Tuple

import numpy as np

from pyhdc.components.binding import ElementMultiplication
from pyhdc.components.bundling import (
    ElementAddition,
    ElementAdditionBipolarThreshold,
    ElementAdditionBits,
    ElementAdditionCut,
)
from pyhdc.components.elements import BernoulliBipolar, UniformBipolar
from pyhdc.components.similarity import CosineSimilarity
from pyhdc.components.thinning import NoThin
from pyhdc.components.unary import IdentityInverse, Negate, SignNormalize
from pyhdc.encodings.base import Encoding
from pyhdc.generation.base import HDCGenerator
from pyhdc.hypervector import EncodingSpec
from pyhdc.types import Backend, Device

# ============================================================================
# Multiply-Add-Permute Encodings
# ============================================================================


class MAP_C(Encoding):
    r"""
    Multiply-Add-Permute encoding with continuous values.

    Uses bipolar values with element-wise multiplication for binding
    and cosine similarity for comparison.

    Args:
        random_choice_range: Optional float (rho). When set, coordinates whose
            \|pre-aggregate sum\| <= rho * sqrt(N/3) are replaced by independent
            Uniform[-1,1] draws during bundling (band randomization).
    """

    def __init__(
        self,
        dimension: int = 10_000,
        backend: Optional[Backend] = None,
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
            normalize_fn=SignNormalize,
            negative_fn=Negate,
        )


class MAP_I(Encoding):
    r"""
    Multiply-Add-Permute encoding with integer values.

    Uses bipolar integer values with element-wise multiplication for binding
    and cosine similarity for comparison.

    Args:
        random_choice_range: Optional float (rho). When set, coordinates whose
            \|pre-aggregate sum\| <= rho * sqrt(N) are replaced by independent
            {-1, +1} draws during bundling (band randomization).
    """

    def __init__(
        self,
        dimension: int = 10_000,
        backend: Optional[Backend] = None,
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
            element_generator=BernoulliBipolar,
            similarity_fn=CosineSimilarity,
            bundling_fn=bundling_fn,
            thinning_fn=NoThin,
            binding_fn=ElementMultiplication,
            unbinding_fn=ElementMultiplication,
            generator_output_type="bits",
            inverse_fn=IdentityInverse,
            normalize_fn=SignNormalize,
            negative_fn=Negate,
        )


# Smallest signed numpy int dtype able to store a k-bit signed value, in order.
_SIGNED_INT_DTYPES = ((8, np.int8), (16, np.int16), (32, np.int32), (64, np.int64))


def _bits_from_mask(mask: Optional[int], bit_width: Optional[int] = None) -> int:
    """Resolve the effective signed bit width ``k`` from a mask or explicit width.

    An explicit ``bit_width`` wins. Otherwise the mask must be a contiguous
    low-bits value ``2**k - 1`` (so its bit length is ``k``), other masks raise and
    point the caller to ``bit_width``. The default ``mask=(2**32) - 1`` resolves to
    ``k = 32``.
    """
    if bit_width is not None:
        k = int(bit_width)
        if k < 1:
            raise ValueError(f"bit_width must be >= 1, got {bit_width}")
        if k > 64:
            raise ValueError(f"bit_width must be <= 64, got {bit_width}")
        return k
    if mask is None:
        raise ValueError("MAP_I_Bits requires a mask or bit_width")
    m = int(mask)
    if m < 1:
        raise ValueError(f"mask must be a positive 2**k - 1 value, got {mask}")
    if (m & (m + 1)) != 0:
        raise ValueError(
            f"mask must have the form 2**k - 1 (contiguous set bits); got {mask}. "
            "Pass bit_width=k for an explicit k-bit limit instead."
        )
    return m.bit_length()  # equals popcount for a 2**k - 1 mask


def _signed_bounds(k: int) -> Tuple[int, int]:
    """Signed two's-complement range for ``k`` bits: ``[-2**(k-1), 2**(k-1) - 1]``.

    For ``k = 32`` this is exactly ``(np.iinfo(np.int32).min, np.iinfo(np.int32).max)``.
    ``k = 1`` gives ``(-1, 0)`` (the signed 1-bit range, not bipolar ``{-1, 1}``).
    """
    return -(2 ** (k - 1)), (2 ** (k - 1)) - 1


def _storage_dtype(k: int):
    """Smallest signed numpy int dtype that holds a ``k``-bit signed value."""
    for bits, dtype in _SIGNED_INT_DTYPES:
        if k <= bits:
            return dtype
    raise ValueError(
        f"bit width k={k} exceeds the 64-bit signed storage ceiling; "
        "use a mask/bit_width of at most 64 bits."
    )


class MAP_I_Bits(Encoding):
    """
    Multiply-Add-Permute encoding with bit-limited integer values.

    Like MAP_I, but the post-bundle sum saturates to a configurable signed bit
    width instead of growing unbounded. The width is taken from ``bit_width`` if
    given, else from ``mask`` (which must have the form ``2**k - 1``), and the
    storage dtype is widened to fit it (int8/int16/int32/int64). The default
    ``mask=(2**32) - 1`` keeps the historical int32 saturation exactly.

    Binding (``bind``/``unbind``, element-wise multiply) is defined on bipolar
    ``{-1, +1}`` operands, where the product stays bipolar. Binding *bundled*
    (saturated, non-bipolar) vectors at a narrow width can overflow the storage
    dtype and wrap, since the multiply is not re-clipped. Bind before bundling, or
    use a wider ``bit_width`` (or the default int32), if you need to bind sums.

    Args:
        mask: A ``2**k - 1`` value selecting the k-bit saturation width
            (default ``(2**32) - 1``, i.e. int32).
        bit_width: Explicit signed bit width ``k``, overrides ``mask`` when set.
    """

    def __init__(
        self,
        dimension: int = 10_000,
        backend: Optional[Backend] = None,
        device: Optional[Device] = None,
        dtype: Optional[Any] = None,
        mask: int = (2**32) - 1,
        bit_width: Optional[int] = None,
        generator: Optional[HDCGenerator] = None,
        similarity_remap: Optional[Callable] = None,
    ) -> None:
        self._mask = mask
        self._bit_width = bit_width
        super().__init__(
            dimension, backend, device, dtype, mask, generator, similarity_remap
        )

    def _get_encoding_spec(self) -> EncodingSpec:
        k = _bits_from_mask(self._mask, self._bit_width)
        min_val, max_val = _signed_bounds(k)
        bundling_fn = partial(ElementAdditionBits, min_val=min_val, max_val=max_val)
        return EncodingSpec(
            dtype=_storage_dtype(k),
            element_generator=BernoulliBipolar,
            similarity_fn=CosineSimilarity,
            bundling_fn=bundling_fn,
            thinning_fn=NoThin,
            binding_fn=ElementMultiplication,
            unbinding_fn=ElementMultiplication,
            mask=self._mask,
            generator_output_type="words",
            inverse_fn=IdentityInverse,
            normalize_fn=SignNormalize,
            negative_fn=Negate,
        )


class MAP_B(Encoding):
    r"""
    Multiply-Add-Permute with bipolar thresholding.

    Uses bipolar values with thresholding during bundling.

    Args:
        random_choice_range: Optional float (rho). When set, coordinates whose
            \|bipolar sum\| <= rho * sqrt(N) are replaced by independent {-1, +1}
            draws during bundling (band randomization).
    """

    def __init__(
        self,
        dimension: int = 10_000,
        backend: Optional[Backend] = None,
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
            element_generator=BernoulliBipolar,
            similarity_fn=CosineSimilarity,
            bundling_fn=bundling_fn,
            thinning_fn=NoThin,
            binding_fn=ElementMultiplication,
            unbinding_fn=ElementMultiplication,
            generator_output_type="bits",
            inverse_fn=IdentityInverse,
            normalize_fn=SignNormalize,
            negative_fn=Negate,
        )
