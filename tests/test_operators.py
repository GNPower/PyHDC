"""Tests for Hypervector operator overloading (2.1.0)."""

import numpy as np
import pytest

import pyhdc

DIM = 256


def test_add_is_bundle():
    # MAP_I bundling randomizes tie coordinates, so seed before each call to
    # compare the operator against the method on equal footing.
    enc = pyhdc.MAP_I(dimension=DIM)
    a, b = enc.generate(), enc.generate()
    np.random.seed(0)
    left = (a + b).data
    np.random.seed(0)
    right = enc.bundle(a, b).data
    np.testing.assert_array_equal(left, right)


def test_mul_is_bind():
    enc = pyhdc.MAP_I(dimension=DIM)
    a, b = enc.generate(), enc.generate()
    np.testing.assert_array_equal((a * b).data, enc.bind(a, b).data)


def test_truediv_is_unbind():
    enc = pyhdc.MAP_I(dimension=DIM)
    a, b = enc.generate(), enc.generate()
    np.testing.assert_array_equal((a / b).data, enc.unbind(a, b).data)


def test_invert_is_inverse():
    enc = pyhdc.BSC(dimension=DIM)
    a = enc.generate()
    np.testing.assert_array_equal((~a).data, a.inverse().data)


def test_shift_operators_are_permute():
    enc = pyhdc.MAP_I(dimension=DIM)
    a = enc.generate()
    np.testing.assert_array_equal((a >> 4).data, a.permute(4).data)
    np.testing.assert_array_equal((a << 4).data, a.permute(-4).data)


def test_mul_per_family_matches_bind():
    for name in ["MAP_C", "MAP_B", "BSC", "FHRR"]:
        enc = getattr(pyhdc, name)(dimension=DIM)
        a, b = enc.generate(), enc.generate()
        np.testing.assert_array_equal((a * b).data, enc.bind(a, b).data)


def test_bind_unbind_roundtrip_via_operators():
    enc = pyhdc.MAP_B(dimension=DIM)
    a, b = enc.generate(), enc.generate()
    np.testing.assert_array_almost_equal(((a * b) / b).data, a.data)


def test_mbat_truediv_unsupported():
    # MBAT unbind needs the binding matrices; the bare operator must error, not
    # silently return a wrong result (same as enc.unbind(a, b) today).
    enc = pyhdc.MBAT(dimension=DIM)
    a, b = enc.generate(), enc.generate()
    with pytest.raises(TypeError):
        _ = a / b


def test_cdt_inverse_and_unbind_raise():
    enc = pyhdc.BSDC_CDT(dimension=DIM)
    a, b = enc.generate(), enc.generate()
    with pytest.raises(NotImplementedError):
        _ = ~a
    with pytest.raises(NotImplementedError):
        _ = a / b


def test_non_hypervector_operand_raises_typeerror():
    enc = pyhdc.MAP_I(dimension=DIM)
    a = enc.generate()
    for bad in (5, "x", None, 2.0):
        with pytest.raises(TypeError):
            _ = a + bad
        with pytest.raises(TypeError):
            _ = a * bad


def test_shift_operand_type_rules():
    enc = pyhdc.MAP_I(dimension=DIM)
    a = enc.generate()
    # numpy integer scalar is accepted
    assert (a >> np.int64(3)).shape == a.shape
    # bool and float are rejected
    with pytest.raises(TypeError):
        _ = a >> True
    with pytest.raises(TypeError):
        _ = a >> 1.5


def test_operator_backend_mismatch_raises():
    if not pyhdc.TORCH_AVAILABLE:
        pytest.skip("PyTorch not installed")
    enc_np = pyhdc.MAP_I(dimension=DIM)
    enc_t = pyhdc.MAP_I(dimension=DIM, backend="torch")
    a, b = enc_np.generate(), enc_t.generate()
    with pytest.raises(ValueError):
        _ = a + b


@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
def test_operators_on_torch_backend():
    import torch

    enc = pyhdc.MAP_I(dimension=128, backend="torch")
    a, b = enc.generate(), enc.generate()
    # bind and permute are deterministic
    assert bool(((a * b).data == enc.bind(a, b).data).all())
    assert bool(((a >> 2).data == a.permute(2).data).all())
    # bundle randomizes ties (torch RNG); seed before each call
    torch.manual_seed(0)
    left = (a + b).data
    torch.manual_seed(0)
    right = enc.bundle(a, b).data
    assert bool((left == right).all())
