"""Tests for batched hypervector manipulation (bundle / bind / select / similarity)."""

import numpy as np
import pytest

import pyhdc
from pyhdc.hypervector import Hypervector

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
]


def make_enc(name):
    cls = getattr(pyhdc, name)
    return cls(dimension=VTB_DIM if name == "VTB" else DIM)


@pytest.mark.parametrize("enc_name", ALL_ENCODINGS)
def test_generate_batch_shape(enc_name):
    enc = make_enc(enc_name)
    batch = enc.generate(size=(enc.dimension, 7))
    assert isinstance(batch, Hypervector)
    assert batch.shape == (enc.dimension, 7)


@pytest.mark.parametrize("enc_name", ALL_ENCODINGS)
def test_bundle_batch_collapses(enc_name):
    enc = make_enc(enc_name)
    batch = enc.generate(size=(enc.dimension, 5))
    bundled = enc.bundle(batch)
    assert isinstance(bundled, Hypervector)
    assert bundled.shape == (enc.dimension,)


@pytest.mark.parametrize("enc_name", ALL_ENCODINGS)
def test_select_columns(enc_name):
    enc = make_enc(enc_name)
    batch = enc.generate(size=(enc.dimension, 6))
    selected = batch.select([0, 2, 4])
    assert selected.shape == (enc.dimension, 3)
    np.testing.assert_array_equal(selected.data, batch.data[:, [0, 2, 4]])


@pytest.mark.parametrize("enc_name", ALL_ENCODINGS)
def test_select_empty(enc_name):
    enc = make_enc(enc_name)
    batch = enc.generate(size=(enc.dimension, 4))
    selected = batch.select([])
    assert selected.shape == (enc.dimension, 0)


@pytest.mark.parametrize("enc_name", ALL_ENCODINGS)
def test_single_batch_similarity_shape(enc_name):
    enc = make_enc(enc_name)
    batch = enc.generate(size=(enc.dimension, 5))
    sims = np.asarray(enc.similarity(batch))
    assert sims.shape == (4,)


def test_bind_two_batches_per_pair_map():
    enc = pyhdc.MAP_I(dimension=128)
    a = enc.generate(size=(128, 4))
    b = enc.generate(size=(128, 4))
    bound = enc.bind(a, b)
    assert bound.shape == (128, 4)
    np.testing.assert_array_equal(bound.data, a.data * b.data)


def test_bind_two_batches_per_pair_bsc():
    enc = pyhdc.BSC(dimension=128)
    a = enc.generate(size=(128, 4))
    b = enc.generate(size=(128, 4))
    bound = enc.bind(a, b)
    assert bound.shape == (128, 4)
    expected = np.logical_xor(a.data, b.data).astype(bound.dtype)
    np.testing.assert_array_equal(bound.data, expected)


def test_bind_unbind_batch_roundtrip_map():
    enc = pyhdc.MAP_B(dimension=128)
    a = enc.generate(size=(128, 4))
    b = enc.generate(size=(128, 4))
    recovered = enc.unbind(enc.bind(a, b), b)
    np.testing.assert_array_almost_equal(recovered.data, a.data)


@pytest.mark.parametrize("enc_name", ["MAP_I", "MAP_C", "MAP_B", "BSC", "HRR"])
def test_bundle_prototype_recalls_member(enc_name):
    enc = make_enc(enc_name)
    np.random.seed(0)
    batch = enc.generate(size=(enc.dimension, 8))
    proto = enc.bundle(batch)
    member = enc.from_array(batch.data[:, 0].copy())
    rand = enc.generate()
    sim_member = float(enc.similarity(proto, member))
    sim_rand = float(enc.similarity(proto, rand))
    assert sim_member > sim_rand


@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTorchBatched:
    def test_batched_pipeline_on_torch(self):
        enc = pyhdc.MAP_I(dimension=128, backend="torch")
        codebook = enc.generate(size=(128, 50))
        assert codebook.backend == "torch"
        assert tuple(codebook.shape) == (128, 50)
        # select with a numpy-array index against a torch tensor
        sample = codebook.select(np.array([0, 5, 10, 20]))
        assert tuple(sample.shape) == (128, 4)
        proto = enc.bundle(sample)
        assert tuple(proto.shape) == (128,)
        sims = enc.similarity(pyhdc.stack([proto, codebook]))
        assert tuple(sims.shape) == (50,)

    def test_bundle_batch_collapse_torch(self):
        enc = pyhdc.MAP_B(dimension=64, backend="torch")
        bundled = enc.bundle(enc.generate(size=(64, 6)))
        assert tuple(bundled.shape) == (64,)

    def test_select_with_list_and_array_torch(self):
        enc = pyhdc.MAP_C(dimension=32, backend="torch")
        batch = enc.generate(size=(32, 8))
        assert tuple(batch.select([1, 3, 5]).shape) == (32, 3)
        assert tuple(batch.select(np.array([1, 3, 5])).shape) == (32, 3)

    def test_bind_two_batches_per_pair_torch(self):
        enc = pyhdc.MAP_I(dimension=64, backend="torch")
        a = enc.generate(size=(64, 4))
        b = enc.generate(size=(64, 4))
        bound = enc.bind(a, b)
        assert tuple(bound.shape) == (64, 4)
        assert bool((bound.data == a.data * b.data).all())


def test_full_bundle_capacity_experiment():
    """The motivating end-to-end experiment runs and recalls above chance."""
    np.random.seed(0)
    enc = pyhdc.MAP_I(dimension=1_000)
    codebook = enc.generate(size=(enc.dimension, 10_000))
    sample_size = 1_000
    idx = np.random.choice(codebook.shape[1], sample_size, replace=False)
    sample = codebook.select(idx)
    proto = enc.bundle(sample)
    sims = enc.similarity(pyhdc.stack([proto, codebook]))
    assert np.asarray(sims).shape == (10_000,)
    top = np.argsort(sims)[-sample_size:]
    accuracy = np.isin(top, idx).mean()
    assert accuracy > sample_size / codebook.shape[1]  # better than chance
