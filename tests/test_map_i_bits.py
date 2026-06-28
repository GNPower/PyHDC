"""Tests for MAP_I_Bits arbitrary-bit-width saturation (the clipping fix)."""

import numpy as np
import pytest

import pyhdc
from pyhdc.encodings.map import _bits_from_mask, _signed_bounds, _storage_dtype


def _bounds(enc):
    kw = enc._spec.bundling_fn.keywords
    return kw["min_val"], kw["max_val"]


def test_default_mask_matches_int32_exactly():
    enc = pyhdc.MAP_I_Bits(dimension=64)
    assert _bounds(enc) == (np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    assert enc._spec.dtype is np.int32


def test_narrow_mask_8bit_saturates():
    enc = pyhdc.MAP_I_Bits(dimension=64, mask=(2**8) - 1)
    assert _bounds(enc) == (-128, 127)
    assert enc._spec.dtype is np.int8
    pos = np.ones((64, 300), dtype=enc._spec.dtype)
    assert int(enc.bundle(pos).data.max()) == 127
    assert int(enc.bundle(-pos).data.min()) == -128


def test_explicit_bit_width_overrides_mask():
    enc = pyhdc.MAP_I_Bits(dimension=64, mask=(2**8) - 1, bit_width=4)
    assert _bounds(enc) == (-8, 7)
    assert enc._spec.dtype is np.int8


def test_wide_bit_width_widens_dtype_no_wrap():
    enc = pyhdc.MAP_I_Bits(dimension=8, bit_width=40)
    assert _bounds(enc) == (-(2**39), 2**39 - 1)
    assert enc._spec.dtype is np.int64


@pytest.mark.parametrize(
    "bad", [{"mask": 10}, {"mask": 0}, {"bit_width": 0}, {"bit_width": 65}]
)
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        pyhdc.MAP_I_Bits(dimension=8, **bad)


def test_helpers():
    assert _bits_from_mask((2**32) - 1) == 32
    assert _bits_from_mask(255) == 8
    assert _bits_from_mask(None, 12) == 12
    assert _signed_bounds(8) == (-128, 127)
    assert _signed_bounds(1) == (-1, 0)
    assert _storage_dtype(8) is np.int8
    assert _storage_dtype(16) is np.int16
    assert _storage_dtype(32) is np.int32
    assert _storage_dtype(40) is np.int64
    with pytest.raises(ValueError):
        _storage_dtype(65)


def test_end_to_end_default():
    enc = pyhdc.MAP_I_Bits(dimension=128)
    a, b = enc.generate(), enc.generate()
    bundled = enc.bundle(a, b)
    assert bundled.shape == (128,)
    assert isinstance(float(enc.similarity(a, bundled)), float)


def test_bipolar_binding_exact_at_narrow_width():
    # Binding bipolar {-1,+1} operands stays bipolar and unbinds exactly, even at
    # a narrow (int8) width.
    enc = pyhdc.MAP_I_Bits(dimension=128, mask=(2**8) - 1)
    assert enc._spec.dtype is np.int8
    a, b = enc.generate(), enc.generate()
    bound = enc.bind(a, b)
    assert set(np.unique(bound.data).tolist()) <= {-1, 1}
    assert np.array_equal(enc.unbind(bound, b).data, a.data)


def test_narrow_width_bundled_binding_overflows_by_design():
    # Documented limitation: binding *bundled* (saturated, non-bipolar) narrow-width
    # vectors overflows the storage dtype (multiply is not re-clipped).
    enc = pyhdc.MAP_I_Bits(dimension=8, bit_width=8)  # int8 storage
    a = enc.from_array(np.full(8, 100, dtype=np.int8))
    b = enc.from_array(np.full(8, 100, dtype=np.int8))
    bound = enc.bind(a, b)
    assert bound.data.dtype == np.int8
    # 100 * 100 = 10000 wraps in int8
    assert int(bound.data[0]) != 10000
