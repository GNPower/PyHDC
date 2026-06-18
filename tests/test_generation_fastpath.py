"""Tests for the vectorized batched-generation fast path (2.1.0).

The fast path must produce results byte-identical to the sequential per-column
loop: ``generate(size=(D, N))`` equals ``N`` successive ``generate(size=D)``
draws under a fixed seed, for every encoding (i.i.d. generators take the fast
path; ordered/segmented generators fall back to the loop).
"""

import random

import numpy as np
import pytest

import pyhdc

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
# Encodings backed by an i.i.d. element generator (SparseSegmented is excluded).
FAST_PATH = ["MAP_C", "MAP_I", "MAP_B", "HRR", "FHRR", "BSC", "BSDC_S"]


def make_enc(name):
    cls = getattr(pyhdc, name)
    return cls(dimension=VTB_DIM if name == "VTB" else DIM)


def _seed(value):
    # Seed both RNGs: i.i.d. generators use numpy, SparseSegmented uses stdlib random.
    np.random.seed(value)
    random.seed(value)


@pytest.mark.parametrize("name", ALL_ENCODINGS)
def test_batch_matches_sequential_2d(name):
    enc = make_enc(name)
    d = enc.dimension
    _seed(123)
    batch = enc.generate(size=(d, N)).data
    _seed(123)
    cols = [enc.generate(size=d).data for _ in range(N)]
    seq = np.stack(cols, axis=-1)
    np.testing.assert_array_equal(batch, seq)


@pytest.mark.parametrize("name", FAST_PATH)
def test_batch_matches_sequential_3d(name):
    enc = make_enc(name)
    d = enc.dimension
    _seed(7)
    b3 = enc.generate(size=(d, 3, 2)).data
    _seed(7)
    cols = [enc.generate(size=d).data for _ in range(6)]
    seq = np.stack(cols, axis=-1).reshape((d, 3, 2))
    np.testing.assert_array_equal(b3, seq)


@pytest.mark.parametrize("name", ALL_ENCODINGS)
def test_fast_path_layout_and_dtype(name):
    enc = make_enc(name)
    d = enc.dimension
    data = enc.generate(size=(d, N)).data
    assert data.flags["C_CONTIGUOUS"]  # matches the 2.0 np.stack(...).reshape layout
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
    def test_torch_batch_matches_sequential(self, name):
        import torch

        cls = getattr(pyhdc, name)
        enc = cls(dimension=DIM, backend="torch")
        _seed(99)
        batch = enc.generate(size=(DIM, N)).data
        _seed(99)
        cols = [enc.generate(size=DIM).data for _ in range(N)]
        seq = torch.stack(cols, dim=-1)
        assert torch.equal(batch, seq)

    def test_torch_fast_path_contiguous(self):
        enc = pyhdc.MAP_I(dimension=DIM, backend="torch")
        assert enc.generate(size=(DIM, N)).data.is_contiguous()
