"""Tests for batched (dimension-first) hypervector generation."""

import numpy as np
import pytest

import pyhdc
from pyhdc.generation.base import DefaultGenerator


def test_batched_generation_shape_and_dtype():
    enc = pyhdc.MAP_C(dimension=64)
    batch = enc.generate(size=(64, 10))
    assert batch.shape == (64, 10)
    assert np.issubdtype(batch.dtype, np.floating)


def test_one_tuple_is_single_vector():
    enc = pyhdc.MAP_I(dimension=40)
    hv = enc.generate(size=(40,))
    assert hv.shape == (40,)


def test_batched_generation_higher_rank():
    enc = pyhdc.MAP_I(dimension=32)
    batch = enc.generate(size=(32, 3, 2))
    assert batch.shape == (32, 3, 2)


def test_batched_generation_matches_sequential_numpy():
    """Under a fixed seed, a (D, N) batch equals N successive (D,) generations."""
    enc = pyhdc.MAP_I(dimension=128)
    np.random.seed(123)
    batch = enc.generate(size=(128, 4)).data
    np.random.seed(123)
    seq = np.stack([enc.generate(size=128).data for _ in range(4)], axis=-1)
    np.testing.assert_array_equal(batch, seq)


def test_batched_generation_matches_sequential_custom_generator():
    """Reproducibility holds for ordered pyhdc generators too."""
    gen = DefaultGenerator(seed=7)
    enc = pyhdc.MAP_I(dimension=64, generator=gen)
    batch = enc.generate(size=(64, 3), use_generator=True).data
    gen.set_seed(7)
    seq = np.stack(
        [enc.generate(size=64, use_generator=True).data for _ in range(3)], axis=-1
    )
    np.testing.assert_array_equal(batch, seq)


@pytest.mark.parametrize(
    "enc_name", ["HRR", "VTB", "MBAT", "BSC", "BSDC_S", "BSDC_SEG", "BSDC_CDT"]
)
def test_batched_generation_works_for_all_generators(enc_name):
    """NormalReal and the sparse generators previously raised on a tuple size."""
    cls = getattr(pyhdc, enc_name)
    dim = 484 if enc_name == "VTB" else 256
    enc = cls(dimension=dim)
    batch = enc.generate(size=(dim, 4))
    assert batch.shape == (dim, 4)
