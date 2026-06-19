"""Tests for first-class permute / inverse / negative / normalize (2.1.0)."""

import numpy as np
import pytest

import pyhdc

DIM = 256
VTB_DIM = 484

ALL_ENCODINGS = [
    "MAP_C",
    "MAP_I",
    "MAP_I_Bits",
    "MAP_B",
    "HRR",
    "HRR_NoNorm",
    "HRR_ConstNorm",
    "FHRR",
    "VTB",
    "MBAT",
    "BSC",
    "BSDC_CDT",
    "BSDC_S",
    "BSDC_SEG",
    "BSDC_THIN",
]

SELF_INVERSE = ["MAP_I", "MAP_I_Bits", "MAP_B", "BSC"]
INVOLUTION = ["HRR", "HRR_NoNorm", "HRR_ConstNorm"]
INVERSE_RAISES = ["MAP_C", "VTB", "MBAT", "BSDC_CDT", "BSDC_S", "BSDC_SEG", "BSDC_THIN"]
NORMALIZE_L2 = ["HRR", "HRR_NoNorm", "HRR_ConstNorm", "VTB", "MBAT"]
NORMALIZE_SIGN = ["MAP_C", "MAP_I", "MAP_I_Bits", "MAP_B"]
NORMALIZE_RAISES = ["BSC", "BSDC_CDT", "BSDC_S", "BSDC_SEG", "BSDC_THIN"]
NEGATE = [
    "MAP_C",
    "MAP_I",
    "MAP_I_Bits",
    "MAP_B",
    "HRR",
    "HRR_NoNorm",
    "HRR_ConstNorm",
    "VTB",
    "MBAT",
]
NEGATE_RAISES = ["FHRR", "BSC", "BSDC_CDT", "BSDC_S", "BSDC_SEG", "BSDC_THIN"]


def make_enc(name):
    cls = getattr(pyhdc, name)
    return cls(dimension=VTB_DIM if name == "VTB" else DIM)


# --------------------------------------------------------------------------- #
# Permute (shared CyclicShift, every encoding)                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_ENCODINGS)
def test_permute_roundtrip(name):
    enc = make_enc(name)
    a = enc.generate()
    assert np.array_equal((a >> 3 << 3).data, a.data)


@pytest.mark.parametrize("name", ["MAP_I", "HRR", "BSC"])
def test_permute_matches_roll(name):
    enc = make_enc(name)
    a = enc.generate()
    np.testing.assert_array_equal(a.permute(5).data, np.roll(a.data, 5, axis=0))


def test_permute_rolls_axis0_of_batch():
    enc = pyhdc.MAP_I(dimension=DIM)
    batch = enc.generate(size=(DIM, 4))
    out = batch.permute(2)
    assert out.shape == (DIM, 4)
    np.testing.assert_array_equal(out.data, np.roll(batch.data, 2, axis=0))


# --------------------------------------------------------------------------- #
# Inverse                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", SELF_INVERSE)
def test_inverse_self_inverse(name):
    enc = make_enc(name)
    a = enc.generate()
    np.testing.assert_array_equal(a.inverse().data, a.data)


@pytest.mark.parametrize("name", INVOLUTION)
def test_inverse_hrr_involution(name):
    enc = make_enc(name)
    a = enc.generate()
    inv = a.inverse()
    assert inv.data[0] == a.data[0]
    np.testing.assert_allclose(inv.data[1:], a.data[1:][::-1])


def test_inverse_fhrr_phase_negate():
    enc = pyhdc.FHRR(dimension=DIM)
    a = enc.generate()
    np.testing.assert_allclose(a.inverse().data, np.mod(-a.data, 2 * np.pi))


@pytest.mark.parametrize("name", INVERSE_RAISES)
def test_inverse_raises_where_undefined(name):
    enc = make_enc(name)
    with pytest.raises(NotImplementedError):
        enc.generate().inverse()


# --------------------------------------------------------------------------- #
# Normalize                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NORMALIZE_L2)
def test_normalize_l2_unit(name):
    enc = make_enc(name)
    out = enc.generate().normalize()
    assert abs(float(np.linalg.norm(out.data)) - 1.0) < 1e-5


@pytest.mark.parametrize("name", NORMALIZE_SIGN)
def test_normalize_sign(name):
    enc = make_enc(name)
    a = enc.bundle(enc.generate(size=(enc.dimension, 5)))
    np.testing.assert_array_equal(a.normalize().data, np.sign(a.data))


def test_normalize_fhrr_wraps_to_canonical_range():
    enc = pyhdc.FHRR(dimension=DIM)
    a = enc.generate()
    out = enc.normalize(a).data
    assert np.all(out >= -np.pi) and np.all(out < np.pi)


@pytest.mark.parametrize("name", NORMALIZE_RAISES)
def test_normalize_raises_where_undefined(name):
    enc = make_enc(name)
    with pytest.raises(NotImplementedError):
        enc.generate().normalize()


# --------------------------------------------------------------------------- #
# Negative                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NEGATE)
def test_negative_negates(name):
    enc = make_enc(name)
    a = enc.generate()
    np.testing.assert_array_equal(a.negative().data, -a.data)


@pytest.mark.parametrize("name", NEGATE_RAISES)
def test_negative_raises_where_undefined(name):
    enc = make_enc(name)
    with pytest.raises(NotImplementedError):
        enc.generate().negative()


# --------------------------------------------------------------------------- #
# Torch parity                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTorchUnary:
    def test_permute_roundtrip_torch(self):
        enc = pyhdc.MAP_I(dimension=128, backend="torch")
        a = enc.generate()
        assert bool(((a >> 3 << 3).data == a.data).all())

    def test_inverse_self_inverse_torch(self):
        enc = pyhdc.BSC(dimension=128, backend="torch")
        a = enc.generate()
        assert bool((a.inverse().data == a.data).all())

    def test_normalize_l2_torch(self):
        import torch

        enc = pyhdc.HRR(dimension=128, backend="torch")
        out = enc.generate().normalize()
        assert abs(float(torch.linalg.norm(out.data)) - 1.0) < 1e-5
