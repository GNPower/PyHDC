"""Tests for the data encoders (pyhdc.encoders)."""

import numpy as np
import pytest

import pyhdc
from pyhdc.hypervector import Hypervector

DIM = 256


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    pyhdc.prefer_numpy()


# ---------------------------------------------------------------- codebook


def test_level_scalar_and_batch():
    enc = pyhdc.Level(pyhdc.MAP_I(dimension=DIM), levels=100, low=0.0, high=1.0)
    hv = enc.encode(0.5)
    assert isinstance(hv, Hypervector) and hv.shape == (DIM,)
    assert hv.encoding.__class__.__name__ == "MAP_I"
    assert enc.encode([0.0, 0.5, 1.0]).shape == (DIM, 3)


def test_level_endpoints_and_clamp():
    enc = pyhdc.Level(pyhdc.MAP_I(dimension=DIM), levels=50, low=0.0, high=1.0)
    np.testing.assert_array_equal(enc.encode(0.0).data, enc.params[:, 0])
    np.testing.assert_array_equal(enc.encode(1.0).data, enc.params[:, -1])
    np.testing.assert_array_equal(enc.encode(-5.0).data, enc.encode(0.0).data)
    np.testing.assert_array_equal(enc.encode(5.0).data, enc.encode(1.0).data)


def test_level_monotone_recall():
    np.random.seed(3)
    base = pyhdc.MAP_I(dimension=8192)
    enc = pyhdc.Level(base, levels=20, low=0.0, high=1.0)
    zero = enc.encode(0.0)
    sims = [
        float(base.similarity(zero.data, enc.encode(x).data))
        for x in np.linspace(0, 1, 20)
    ]
    assert all(sims[i] >= sims[i + 1] - 1e-9 for i in range(19))


def test_circular_wraps():
    enc = pyhdc.Circular(pyhdc.MAP_I(dimension=DIM), levels=12, low=0.0, high=12.0)
    np.testing.assert_array_equal(enc.encode(12.0).data, enc.encode(0.0).data)
    np.testing.assert_array_equal(enc.encode(-1.0).data, enc.encode(11.0).data)


@pytest.mark.parametrize("cls", [pyhdc.Empty, pyhdc.Random])
def test_trivial_codebook_shapes(cls):
    enc = cls(pyhdc.MAP_I(dimension=DIM), levels=10)
    assert enc.encode(3).shape == (DIM,)
    assert enc.encode([1, 2]).shape == (DIM, 2)


def test_identity_is_binding_identity():
    base = pyhdc.MAP_I(dimension=DIM)
    enc = pyhdc.Identity(base, levels=8)
    e = enc.encode(3)
    assert e.shape == (DIM,)
    x = base.generate()
    bound = base.bind(x, e)  # bind(x, identity) == x
    assert float(base.similarity(bound, x)) > 0.99
    assert enc.encode([1, 2]).shape == (DIM, 2)


@pytest.mark.parametrize("name", ["VTB", "MBAT", "BSDC_S", "BSDC_CDT"])
def test_identity_unsupported_raises(name):
    kw = {"dimension": 484} if name == "VTB" else {"dimension": DIM}
    with pytest.raises(NotImplementedError):
        pyhdc.Identity(getattr(pyhdc, name)(**kw), levels=8)


def test_empty_encodes_zero():
    enc = pyhdc.Empty(pyhdc.MAP_I(dimension=DIM), levels=10)
    assert np.all(enc.encode(3).data == 0)


def test_levels_validation():
    with pytest.raises(ValueError):
        pyhdc.Level(pyhdc.MAP_I(dimension=DIM), levels=0)
    with pytest.raises(ValueError):
        pyhdc.Level(pyhdc.MAP_I(dimension=DIM), levels=10, low=1.0, high=1.0)


@pytest.mark.parametrize("name", ["MAP_I", "MAP_B", "BSC"])
def test_thermometer_discrete_ok(name):
    enc = pyhdc.Thermometer(getattr(pyhdc, name)(dimension=DIM), levels=10)
    assert enc.encode(0.5).shape == (DIM,)


@pytest.mark.parametrize("name", ["MAP_C", "HRR", "VTB", "MBAT", "FHRR"])
def test_thermometer_continuous_raises(name):
    kw = {"dimension": 484} if name == "VTB" else {"dimension": DIM}
    with pytest.raises(NotImplementedError):
        pyhdc.Thermometer(getattr(pyhdc, name)(**kw), levels=10)


# ---------------------------------------------------------------- functional


def test_projection_maps_into_bipolar_domain():
    enc = pyhdc.Projection(pyhdc.MAP_I(dimension=DIM), features=8)
    out = enc.encode(np.random.randn(8))
    assert out.shape == (DIM,)
    assert set(np.unique(out.data).tolist()) <= {-1, 0, 1}  # SignNormalize
    assert enc.encode(np.random.randn(8, 5)).shape == (DIM, 5)


def test_projection_hrr_real():
    enc = pyhdc.Projection(pyhdc.HRR(dimension=DIM), features=8)
    assert enc.encode(np.random.randn(8)).shape == (DIM,)


@pytest.mark.parametrize("name", ["BSC", "BSDC_S"])
def test_projection_unsupported_raises(name):
    with pytest.raises(NotImplementedError):
        pyhdc.Projection(getattr(pyhdc, name)(dimension=DIM), features=8)


def test_sinusoid_bounded():
    enc = pyhdc.Sinusoid(pyhdc.HRR(dimension=DIM), features=8)
    out = enc.encode(np.random.randn(8))
    assert out.shape == (DIM,)
    assert float(np.abs(out.data).max()) <= np.sqrt(2.0 / DIM) + 1e-6


def test_density_discrete_only():
    enc = pyhdc.Density(pyhdc.MAP_I(dimension=DIM))
    assert enc.encode(0.5).shape == (DIM,)
    assert enc.encode([0.1, 0.9]).shape == (DIM, 2)
    with pytest.raises(NotImplementedError):
        pyhdc.Density(pyhdc.HRR(dimension=DIM))


def test_fractionalpower_fhrr():
    enc = pyhdc.FractionalPower(pyhdc.FHRR(dimension=DIM))
    assert np.allclose(enc.encode(1.0).data, enc.params, atol=1e-5)
    assert np.allclose(enc.encode(0.0).data, 0.0, atol=1e-6)


def test_fractionalpower_hrr_roundtrip():
    enc = pyhdc.FractionalPower(pyhdc.HRR(dimension=DIM))
    assert np.allclose(enc.encode(1.0).data, enc.params, atol=1e-4)


@pytest.mark.parametrize("name", ["MAP_I", "BSC", "VTB"])
def test_fractionalpower_unsupported_raises(name):
    kw = {"dimension": 484} if name == "VTB" else {"dimension": DIM}
    with pytest.raises(NotImplementedError):
        pyhdc.FractionalPower(getattr(pyhdc, name)(**kw))


# ---------------------------------------------------------------- torch parity


@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
def test_torch_parity():
    lvl = pyhdc.Level(
        pyhdc.MAP_I(dimension=DIM, backend="torch"), levels=20, low=0, high=1
    )
    hv = lvl.encode([0.0, 1.0])
    assert hv.backend == "torch" and tuple(hv.shape) == (DIM, 2)
    proj = pyhdc.Projection(pyhdc.MAP_I(dimension=DIM, backend="torch"), features=8)
    assert proj.encode(np.random.randn(8)).backend == "torch"
    fp = pyhdc.FractionalPower(pyhdc.FHRR(dimension=DIM, backend="torch"))
    assert tuple(fp.encode(2.0).shape) == (DIM,)
