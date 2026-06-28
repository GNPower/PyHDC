"""Tests for the family-aware basis builders (pyhdc.components.basis)."""

import numpy as np
import pytest

import pyhdc
from pyhdc.components import basis

DIM = 256


def test_random_shape_and_bipolar_domain():
    enc = pyhdc.MAP_I(dimension=DIM)
    r = basis.random(enc, 5)
    assert r.shape == (DIM, 5)
    assert set(np.unique(r).tolist()) <= {-1, 1}


def test_random_binary_domain():
    enc = pyhdc.BSC(dimension=DIM)
    assert set(np.unique(basis.random(enc, 4)).tolist()) <= {0, 1}


def test_empty_is_zeros():
    enc = pyhdc.MAP_I(dimension=DIM)
    e = basis.empty(enc, 3)
    assert e.shape == (DIM, 3)
    assert np.all(e == 0)


def test_dimension_override():
    enc = pyhdc.MAP_I(dimension=DIM)
    assert basis.random(enc, 2, dimension=64).shape == (64, 2)


def test_level_monotone_and_endpoints():
    np.random.seed(1)
    enc = pyhdc.MAP_I(dimension=8192)
    lv = basis.level(enc, 12)
    sims = [float(enc.similarity(lv[:, 0], lv[:, i])) for i in range(12)]
    assert sims[0] == pytest.approx(1.0, abs=1e-6)
    assert all(sims[i] >= sims[i + 1] - 1e-9 for i in range(11))  # non-increasing
    assert abs(sims[-1]) < 0.2  # ends near-orthogonal


def test_circular_wraps_and_dips_at_half_ring():
    np.random.seed(2)
    enc = pyhdc.MAP_I(dimension=8192)
    cz = basis.circular(enc, 12)
    sims = [float(enc.similarity(cz[:, 0], cz[:, i])) for i in range(12)]
    assert sims[1] == pytest.approx(sims[11], abs=0.1)  # ring adjacency
    assert int(np.argmin(sims)) in (5, 6, 7)  # min near the half ring


def test_thermometer_nested_and_endpoints():
    enc = pyhdc.MAP_I(dimension=DIM)
    th = basis.thermometer(enc, 8)
    assert np.all(th[:, 0] == -1) and np.all(th[:, -1] == 1)
    for i in range(1, 8):
        prev = set(np.where(th[:, i - 1] == 1)[0])
        cur = set(np.where(th[:, i] == 1)[0])
        assert prev.issubset(cur)


@pytest.mark.parametrize("name", ["MAP_C", "HRR", "HRR_NoNorm", "FHRR", "MBAT"])
def test_thermometer_discrete_only(name):
    with pytest.raises(NotImplementedError):
        basis.thermometer(getattr(pyhdc, name)(dimension=DIM), 5)


def test_family_endpoints():
    assert basis.family_endpoints(pyhdc.MAP_I(dimension=DIM)) == (-1, 1)
    assert basis.family_endpoints(pyhdc.BSC(dimension=DIM)) == (0, 1)
    assert basis.family_endpoints(pyhdc.BSDC_S(dimension=DIM)) == (0, 1)
    with pytest.raises(NotImplementedError):
        basis.family_endpoints(pyhdc.MAP_C(dimension=DIM))


@pytest.mark.parametrize("name", ["MAP_C", "MAP_I", "MAP_B", "HRR", "FHRR", "BSC"])
def test_binding_identity_roundtrip(name):
    enc = getattr(pyhdc, name)(dimension=DIM)
    e = basis.binding_identity(enc)
    assert e.shape == (DIM,)
    x = enc.generate()
    bound = enc.bind(x, enc.from_array(e))  # bind(x, e) == x
    assert float(enc.similarity(bound, x)) > 0.99


def test_identity_builder_tiles_the_element():
    enc = pyhdc.MAP_I(dimension=DIM)
    cb = basis.identity(enc, 5)
    assert cb.shape == (DIM, 5)
    e = basis.binding_identity(enc)
    assert np.array_equal(cb[:, 0], e) and np.array_equal(cb[:, 4], e)


@pytest.mark.parametrize("name", ["VTB", "MBAT", "BSDC_CDT", "BSDC_S", "BSDC_SEG"])
def test_binding_identity_unsupported_raises(name):
    kw = {"dimension": 484} if name == "VTB" else {"dimension": DIM}
    with pytest.raises(NotImplementedError):
        basis.binding_identity(getattr(pyhdc, name)(**kw))


@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
def test_basis_torch_backend():
    enc = pyhdc.MAP_I(dimension=DIM, backend="torch")
    lv = basis.level(enc, 6)
    assert tuple(lv.shape) == (DIM, 6)
    th = basis.thermometer(enc, 6)
    assert set(th.unique().tolist()) <= {-1, 1}
    em = basis.empty(enc, 4)  # exercises the zeros torch path
    assert tuple(em.shape) == (DIM, 4) and bool((em == 0).all())
    idt = basis.identity(enc, 4)
    assert bool((idt == 1).all())  # MAP multiply identity is all-ones
