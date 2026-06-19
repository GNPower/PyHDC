"""Tests for (D, N, M) axis-aware bundle / similarity / bind (2.1.0)."""

import numpy as np
import pytest

import pyhdc
from pyhdc.components.bundling import (
    DisjunctionThinned,
    ElementAddition,
    ElementAdditionBits,
)

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
def test_non_batch_safe_bind_auto_batches(name):
    # Non-batch-safe binders are applied per column internally, so a batched bind
    # returns one batched Hypervector instead of raising.
    enc = make_enc(name)
    d = enc.dimension
    a = enc.generate(size=(d, 4))
    b = enc.generate(size=(d, 4))
    bound = enc.bind(a, b)
    assert tuple(bound.shape) == (d, 4)


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

    def test_convolution_auto_batches_torch(self):
        enc = pyhdc.HRR(dimension=128, backend="torch")
        a = enc.generate(size=(128, 4))
        b = enc.generate(size=(128, 4))
        assert tuple(enc.bind(a, b).shape) == (128, 4)


# --------------------------------------------------------------------------- #
# ElementAdditionBits: sum then clip once, saturating at the bounds            #
# --------------------------------------------------------------------------- #
def _ref_bits(batch, lo, hi):
    # Reference: accumulate in a wide integer, then clip once.
    return np.clip(batch.sum(axis=1, dtype=np.int64), lo, hi).astype(batch.dtype)


def test_element_addition_bits_wide_bounds_is_plain_sum():
    rng = np.random.RandomState(0)
    batch = rng.randint(-1, 2, size=(64, 40)).astype(np.int32)
    lo, hi = int(np.iinfo(np.int32).min), int(np.iinfo(np.int32).max)
    np.testing.assert_array_equal(
        ElementAdditionBits(batch, min_val=lo, max_val=hi), _ref_bits(batch, lo, hi)
    )


def test_element_addition_bits_saturates_at_bounds():
    # Tight bounds clip the final sum once (saturate), not per addition.
    rng = np.random.RandomState(1)
    batch = rng.randint(-1, 2, size=(64, 40)).astype(np.int32)
    out = ElementAdditionBits(batch, min_val=-1, max_val=1)
    np.testing.assert_array_equal(out, np.clip(batch.sum(axis=1), -1, 1))


def test_element_addition_bits_no_int32_overflow():
    # Many large positives would overflow int32 if accumulated narrow, the wide
    # accumulator + final clip must saturate at max_val, not wrap to negative.
    big = int(np.iinfo(np.int32).max)
    batch = np.full((4, 8), big, dtype=np.int32)
    out = ElementAdditionBits(batch, min_val=int(np.iinfo(np.int32).min), max_val=big)
    np.testing.assert_array_equal(out, np.full(4, big, dtype=np.int32))


# --------------------------------------------------------------------------- #
# batch_dim is vectorized (no per-group Python loop) and equals the axis split  #
# --------------------------------------------------------------------------- #
def test_batch_dim_vectorized_matches_axis_split():
    # MAP_I_Bits bundling is deterministic, so the (deprecated) batch_dim split
    # must equal the axis= reduction split column-for-column.
    enc = pyhdc.MAP_I_Bits(dimension=128)
    arr = np.random.RandomState(0).randint(-1, 2, size=(128, 8, 20)).astype(np.int32)
    with pytest.warns(DeprecationWarning):
        lst = enc.bundle(arr, batch_dim=2)
    assert isinstance(lst, list) and len(lst) == 20 and lst[0].shape == (128,)
    got = np.stack([h.data for h in lst], axis=1)
    np.testing.assert_array_equal(got, enc.bundle(arr, axis=1).data)
    # batch_dim=1 keeps axis 1 (8 results)
    with pytest.warns(DeprecationWarning):
        assert len(enc.bundle(arr, batch_dim=1)) == 8


@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
def test_batch_dim_vectorized_torch():
    import torch

    enc = pyhdc.MAP_I_Bits(dimension=64, backend="torch")
    arr = enc.generate(size=(64, 5, 12))
    with pytest.warns(DeprecationWarning):
        lst = enc.bundle(arr, batch_dim=2)
    assert isinstance(lst, list) and len(lst) == 12
    got = torch.stack([h.data for h in lst], dim=1)
    assert bool((got == enc.bundle(arr, axis=1).data).all())


# --------------------------------------------------------------------------- #
# DisjunctionThinned: batched (rank>1) thinning is vectorized and per-column   #
# --------------------------------------------------------------------------- #
def test_disjunction_thinned_batched_density():
    dim, m = 200, 6
    rng = np.random.RandomState(0)
    # Dense-ish OR bundle so thinning engages on every column.
    batch = (rng.random((dim, 8, m)) < 0.6).astype(np.int8)
    out = DisjunctionThinned(batch, density=0.25, axis=1)
    assert out.shape == (dim, m)
    assert set(np.unique(out)).issubset({0, 1})
    num_nonzero = int(np.ceil(dim * 0.25))
    per_col = (out != 0).sum(axis=0)
    # Each column is thinned to exactly num_nonzero set bits (bundle was dense).
    np.testing.assert_array_equal(per_col, np.full(m, num_nonzero))


def test_disjunction_thinned_batched_reproducible():
    dim, m = 128, 4
    batch = (np.random.RandomState(1).random((dim, 6, m)) < 0.7).astype(np.int8)
    np.random.seed(42)
    a = DisjunctionThinned(batch, density=0.3, axis=1)
    np.random.seed(42)
    b = DisjunctionThinned(batch, density=0.3, axis=1)
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
def test_disjunction_thinned_batched_density_torch():
    import torch

    dim, m = 200, 6
    torch.manual_seed(0)
    batch = (torch.rand(dim, 8, m) < 0.6).to(torch.int8)
    out = DisjunctionThinned(batch, density=0.25, axis=1)
    assert tuple(out.shape) == (dim, m)
    num_nonzero = int(np.ceil(dim * 0.25))
    per_col = (out != 0).sum(dim=0)
    assert bool((per_col == num_nonzero).all())


def test_convolution_bind_unbind_batched_roundtrip():
    # HRR convolution/correlation (non-batch-safe) auto-loops per column, a
    # batched bind then unbind recovers each column approximately.
    enc = pyhdc.HRR(dimension=256)
    a = enc.generate(size=(256, 5))
    b = enc.generate(size=(256, 5))
    bound = enc.bind(a, b)
    assert tuple(bound.shape) == (256, 5)
    recovered = enc.unbind(bound, b)
    for i in range(5):
        sim = enc.similarity(
            enc.from_array(recovered.data[:, i]), enc.from_array(a.data[:, i])
        )
        assert float(sim) > 0.3


def test_batch_dim_emits_deprecation_warning():
    enc = pyhdc.MAP_I(dimension=64)
    arr = enc.generate(size=(64, 3, 4))
    with pytest.warns(DeprecationWarning):
        enc.bundle(arr, batch_dim=2)
