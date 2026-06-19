"""Tests for the vectorized batched-generation fast path (2.1.0).

The i.i.d. fast path draws the whole ``(D, *batch)`` array in one call, so it is
fast and reproducible under a fixed seed for a given shape, but it is NOT
value-identical to generating the vectors one at a time. Ordered/segmented and
custom generators fall back to the per-vector loop and so still match their
sequential output.
"""

import random

import numpy as np
import pytest

import pyhdc
from pyhdc.encodings.base import _IID_ELEMENT_GENERATORS

DIM = 128
VTB_DIM = 484
N = 5

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
# A representative set of encodings backed by an i.i.d. element generator.
FAST_PATH = ["MAP_C", "MAP_I", "MAP_B", "HRR", "FHRR", "BSC", "BSDC_S"]


def make_enc(name):
    cls = getattr(pyhdc, name)
    return cls(dimension=VTB_DIM if name == "VTB" else DIM)


def _seed(value):
    # Seed both RNGs: i.i.d. generators use numpy, SparseSegmented uses stdlib random.
    np.random.seed(value)
    random.seed(value)


def _uses_loop(enc):
    # The fast path returns None (loop fallback) for non-i.i.d. element generators.
    return enc._spec.element_generator not in _IID_ELEMENT_GENERATORS


@pytest.mark.parametrize("name", ALL_ENCODINGS)
def test_batch_reproducible_with_itself(name):
    # Same seed + same shape reproduces the same batch, fast path or loop.
    enc = make_enc(name)
    d = enc.dimension
    _seed(123)
    a = enc.generate(size=(d, N)).data
    _seed(123)
    b = enc.generate(size=(d, N)).data
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", FAST_PATH)
def test_batch_3d_reproducible(name):
    enc = make_enc(name)
    d = enc.dimension
    _seed(7)
    a = enc.generate(size=(d, 3, 2)).data
    _seed(7)
    b = enc.generate(size=(d, 3, 2)).data
    np.testing.assert_array_equal(a, b)
    assert a.shape == (d, 3, 2)


@pytest.mark.parametrize("name", ALL_ENCODINGS)
def test_loop_generators_match_sequential(name):
    # Generators that fall back to the per-vector loop still equal N (D,) draws.
    enc = make_enc(name)
    if not _uses_loop(enc):
        pytest.skip("uses the vectorized fast path, not the sequential loop")
    d = enc.dimension
    _seed(7)
    batch = enc.generate(size=(d, N)).data
    _seed(7)
    cols = [enc.generate(size=d).data for _ in range(N)]
    seq = np.stack(cols, axis=-1)
    np.testing.assert_array_equal(batch, seq)


@pytest.mark.parametrize("name", ALL_ENCODINGS)
def test_fast_path_layout_and_dtype(name):
    enc = make_enc(name)
    d = enc.dimension
    data = enc.generate(size=(d, N)).data
    assert data.flags["C_CONTIGUOUS"]
    assert data.shape == (d, N)
    assert data.dtype == enc.generate().data.dtype


def test_custom_generator_falls_back_and_reproduces():
    from pyhdc.generation.base import DefaultGenerator

    enc = pyhdc.MAP_C(dimension=DIM, generator=DefaultGenerator(seed=5))
    batch = enc.generate(size=(DIM, 3)).data

    enc2 = pyhdc.MAP_C(dimension=DIM, generator=DefaultGenerator(seed=5))
    cols = [enc2.generate(size=DIM).data for _ in range(3)]
    seq = np.stack(cols, axis=-1)
    np.testing.assert_array_equal(batch, seq)


@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTorchFastPath:
    @pytest.mark.parametrize("name", ["MAP_I", "MAP_C", "HRR", "BSC"])
    def test_torch_batch_reproducible(self, name):
        import torch

        cls = getattr(pyhdc, name)
        enc = cls(dimension=DIM, backend="torch")
        _seed(99)
        a = enc.generate(size=(DIM, N)).data
        _seed(99)
        b = enc.generate(size=(DIM, N)).data
        assert torch.equal(a, b)

    def test_torch_fast_path_contiguous(self):
        enc = pyhdc.MAP_I(dimension=DIM, backend="torch")
        assert enc.generate(size=(DIM, N)).data.is_contiguous()
