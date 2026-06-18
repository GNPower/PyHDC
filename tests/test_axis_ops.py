"""Tests for (D, N, M) axis-aware bundle / similarity / bind (2.1.0)."""

import numpy as np
import pytest

import pyhdc
from pyhdc.components.bundling import ElementAddition, ElementAdditionBits

DIM = 256
VTB_DIM = 484

# Encodings whose binding is pure element-wise (batch-safe broadcasting).
ELEMENTWISE = ["MAP_C", "MAP_I", "MAP_B", "BSC", "FHRR"]
# Encodings whose binding is NOT batch-safe (must raise on batched input).
NON_BATCH_SAFE = [
    "HRR",
    "HRR_NoNorm",
    "HRR_ConstNorm",
    "VTB",
    "MBAT",
    "BSDC_S",
    "BSDC_SEG",
    "BSDC_CDT",
]
BUNDLE_SHAPE = ["MAP_C", "MAP_I", "MAP_B", "HRR", "FHRR", "BSC", "BSDC_S"]


def make_enc(name):
    cls = getattr(pyhdc, name)
    return cls(dimension=VTB_DIM if name == "VTB" else DIM)


# --------------------------------------------------------------------------- #
# Generation                                                                   #
# --------------------------------------------------------------------------- #
def test_generate_dnm_shape():
    enc = pyhdc.MAP_I(dimension=DIM)
    batch = enc.generate(size=(DIM, 4, 3))
    assert batch.shape == (DIM, 4, 3)


# --------------------------------------------------------------------------- #
# Bundling                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", BUNDLE_SHAPE)
def test_bundle_dnm_default_and_axes(name):
    enc = make_enc(name)
    d = enc.dimension
    batch = enc.generate(size=(d, 4, 3))
    assert enc.bundle(batch).shape == (d, 4)  # default == last batch axis
    assert enc.bundle(batch, axis=2).shape == (d, 4)
    assert enc.bundle(batch, axis=1).shape == (d, 3)


@pytest.mark.parametrize("name", ["MAP_C", "MAP_I", "BSC", "HRR"])
def test_bundle_tuple_axis_collapses_all(name):
    enc = make_enc(name)
    d = enc.dimension
    batch = enc.generate(size=(d, 4, 3))
    assert enc.bundle(batch, axis=(1, 2)).shape == (d,)


def test_bundle_axis0_rejected():
    enc = pyhdc.MAP_I(dimension=DIM)
    batch = enc.generate(size=(DIM, 4, 3))
    with pytest.raises(ValueError):
        enc.bundle(batch, axis=0)


def test_bundle_axis_and_batch_dim_mutually_exclusive():
    enc = pyhdc.MAP_I(dimension=DIM)
    batch = enc.generate(size=(DIM, 4, 3))
    with pytest.raises(ValueError):
        enc.bundle(batch, axis=1, batch_dim=2)


def test_bundle_dn_backward_compat():
    enc = pyhdc.MAP_I(dimension=DIM)
    batch = enc.generate(size=(DIM, 5))
    assert enc.bundle(batch).shape == (DIM,)
    assert enc.bundle(batch, axis=1).shape == (DIM,)


def test_bundle_axis_value_matches_sum():
    # All-ones has no zero-sum coordinates, so band randomization never triggers
    # and the additive bundle equals a plain sum over the reduce axis.
    arr = np.ones((8, 4, 3), dtype=np.int32)
    res2, _ = ElementAddition(arr, axis=2)
    np.testing.assert_array_equal(res2, np.full((8, 4), 3))
    res1, _ = ElementAddition(arr, axis=1)
    np.testing.assert_array_equal(res1, np.full((8, 3), 4))
    resb, _ = ElementAddition(arr, axis=(1, 2))
    np.testing.assert_array_equal(resb, np.full((8,), 12))


def test_bundle_metadata_rank_gated():
    # (D,) result -> scalar int count; (D, M) result -> per-output count array.
    arr = np.zeros((8, 4, 3), dtype=np.int32)  # all zero -> every coord in band
    _, meta_1d = ElementAddition(arr, axis=(1, 2))
    assert isinstance(meta_1d["random_zone_count"], int)
    _, meta_2d = ElementAddition(arr, axis=2)
    assert np.asarray(meta_2d["random_zone_count"]).shape == (4,)


# --------------------------------------------------------------------------- #
# Similarity                                                                   #
# --------------------------------------------------------------------------- #
def test_similarity_dnm_requires_axis():
    enc = pyhdc.MAP_I(dimension=DIM)
    batch = enc.generate(size=(DIM, 4, 3))
    with pytest.raises(ValueError):
        enc.similarity(batch)


def test_similarity_dnm_axis_shapes():
    enc = pyhdc.MAP_I(dimension=DIM)
    batch = enc.generate(size=(DIM, 4, 3))
    assert np.asarray(enc.similarity(batch, axis=2)).shape == (4, 2)
    assert np.asarray(enc.similarity(batch, axis=1)).shape == (3, 3)


def test_similarity_two_batch_broadcast():
    enc = pyhdc.MAP_I(dimension=DIM)
    a = enc.generate(size=(DIM, 4))
    b = enc.generate(size=(DIM, 4, 3))
    assert np.asarray(enc.similarity(a, b)).shape == (4, 3)


def test_similarity_backward_compat():
    enc = pyhdc.MAP_I(dimension=DIM)
    a = enc.generate()
    b = enc.generate()
    assert isinstance(float(enc.similarity(a, b)), float)
    assert abs(float(enc.similarity(a, a)) - 1.0) < 1e-4
    batch = enc.generate(size=(DIM, 5))
    assert np.asarray(enc.similarity(batch)).shape == (4,)


# --------------------------------------------------------------------------- #
# Binding                                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ELEMENTWISE)
def test_bind_dnm_per_element(name):
    enc = make_enc(name)
    d = enc.dimension
    a = enc.generate(size=(d, 2, 3))
    b = enc.generate(size=(d, 2, 3))
    assert enc.bind(a, b).shape == (d, 2, 3)


def test_bind_broadcast_single_key_map():
    enc = pyhdc.MAP_I(dimension=DIM)
    key = enc.generate()
    batch = enc.generate(size=(DIM, 4))
    bound = enc.bind(key, batch)
    assert bound.shape == (DIM, 4)
    np.testing.assert_array_equal(bound.data, key.data[:, None] * batch.data)


def test_bind_broadcast_dnm_key_map():
    enc = pyhdc.MAP_I(dimension=DIM)
    key = enc.generate()
    batch = enc.generate(size=(DIM, 2, 3))
    bound = enc.bind(key, batch)
    assert bound.shape == (DIM, 2, 3)
    np.testing.assert_array_equal(bound.data, key.data[:, None, None] * batch.data)


@pytest.mark.parametrize("name", NON_BATCH_SAFE)
def test_non_batch_safe_bind_raises_on_batch(name):
    enc = make_enc(name)
    d = enc.dimension
    a = enc.generate(size=(d, 4))
    b = enc.generate(size=(d, 4))
    with pytest.raises(ValueError):
        enc.bind(a, b)


# --------------------------------------------------------------------------- #
# Torch parity                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTorchAxis:
    def test_bundle_dnm_torch(self):
        enc = pyhdc.MAP_I(dimension=128, backend="torch")
        batch = enc.generate(size=(128, 4, 3))
        assert tuple(enc.bundle(batch).shape) == (128, 4)
        assert tuple(enc.bundle(batch, axis=1).shape) == (128, 3)
        assert tuple(enc.bundle(batch, axis=(1, 2)).shape) == (128,)

    def test_bind_broadcast_torch(self):
        enc = pyhdc.MAP_I(dimension=128, backend="torch")
        key = enc.generate()
        batch = enc.generate(size=(128, 4))
        bound = enc.bind(key, batch)
        assert tuple(bound.shape) == (128, 4)
        assert bool((bound.data == key.data[:, None] * batch.data).all())

    def test_similarity_dnm_torch(self):
        enc = pyhdc.MAP_I(dimension=128, backend="torch")
        batch = enc.generate(size=(128, 4, 3))
        assert tuple(enc.similarity(batch, axis=2).shape) == (4, 2)

    def test_convolution_raises_on_batch_torch(self):
        enc = pyhdc.HRR(dimension=128, backend="torch")
        a = enc.generate(size=(128, 4))
        b = enc.generate(size=(128, 4))
        with pytest.raises(ValueError):
            enc.bind(a, b)


# --------------------------------------------------------------------------- #
# ElementAdditionBits: vectorized fast path == sequential per-step clip        #
# --------------------------------------------------------------------------- #
def _ref_bits(batch, lo, hi):
    res = np.zeros(batch.shape[0], dtype=batch.dtype)
    for j in range(batch.shape[1]):
        res = np.clip(res + batch[:, j], lo, hi)
    return res


def test_element_addition_bits_fastpath_matches_loop():
    rng = np.random.RandomState(0)
    batch = rng.randint(-1, 2, size=(64, 40)).astype(np.int32)
    lo, hi = int(np.iinfo(np.int32).min), int(np.iinfo(np.int32).max)
    np.testing.assert_array_equal(
        ElementAdditionBits(batch, min_val=lo, max_val=hi), _ref_bits(batch, lo, hi)
    )


def test_element_addition_bits_tight_bounds_match_loop():
    # Tight bounds force the per-step clip (fallback loop); it must match the
    # reference and differ from a naive sum-then-clip.
    rng = np.random.RandomState(1)
    batch = rng.randint(-1, 2, size=(64, 40)).astype(np.int32)
    out = ElementAdditionBits(batch, min_val=-1, max_val=1)
    np.testing.assert_array_equal(out, _ref_bits(batch, -1, 1))
    assert not np.array_equal(out, np.clip(batch.sum(axis=1), -1, 1))


# --------------------------------------------------------------------------- #
# batch_dim is vectorized (no per-group Python loop) and equals the axis split  #
# --------------------------------------------------------------------------- #
def test_batch_dim_vectorized_matches_axis_split():
    # MAP_I_Bits bundling is deterministic, so the batch_dim split must equal the
    # axis= reduction split column-for-column.
    enc = pyhdc.MAP_I_Bits(dimension=128)
    arr = np.random.RandomState(0).randint(-1, 2, size=(128, 8, 20)).astype(np.int32)
    lst = enc.bundle(arr, batch_dim=2)
    assert isinstance(lst, list) and len(lst) == 20 and lst[0].shape == (128,)
    got = np.stack([h.data for h in lst], axis=1)
    np.testing.assert_array_equal(got, enc.bundle(arr, axis=1).data)
    # batch_dim=1 keeps axis 1 (8 results)
    assert len(enc.bundle(arr, batch_dim=1)) == 8


@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
def test_batch_dim_vectorized_torch():
    import torch

    enc = pyhdc.MAP_I_Bits(dimension=64, backend="torch")
    arr = enc.generate(size=(64, 5, 12))
    lst = enc.bundle(arr, batch_dim=2)
    assert isinstance(lst, list) and len(lst) == 12
    got = torch.stack([h.data for h in lst], dim=1)
    assert bool((got == enc.bundle(arr, axis=1).data).all())
