"""Tests for the backend-agnostic pyhdc.stack."""

import numpy as np
import pytest

import pyhdc


def test_stack_vector_and_batch():
    enc = pyhdc.MAP_I(dimension=64)
    proto = enc.generate(size=64)
    codebook = enc.generate(size=(64, 10))
    combined = pyhdc.stack([proto, codebook])
    assert combined.shape == (64, 11)
    np.testing.assert_array_equal(combined.data[:, 0], proto.data)
    np.testing.assert_array_equal(combined.data[:, 1:], codebook.data)


def test_stack_two_vectors():
    enc = pyhdc.MAP_I(dimension=64)
    a = enc.generate(size=64)
    b = enc.generate(size=64)
    combined = pyhdc.stack([a, b])
    assert combined.shape == (64, 2)


def test_stack_two_batches():
    enc = pyhdc.MAP_C(dimension=48)
    a = enc.generate(size=(48, 3))
    b = enc.generate(size=(48, 5))
    combined = pyhdc.stack([a, b])
    assert combined.shape == (48, 8)


def test_stack_empty_raises():
    with pytest.raises(ValueError):
        pyhdc.stack([])


def test_stack_preserves_encoding_and_backend():
    enc = pyhdc.MAP_C(dimension=32)
    a = enc.generate(size=32)
    b = enc.generate(size=(32, 3))
    combined = pyhdc.stack([a, b])
    assert combined.backend == "numpy"
    assert combined.encoding is enc


@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
def test_stack_backend_mismatch_raises():
    enc = pyhdc.MAP_I(dimension=32)
    a = enc.generate(size=32)
    b = enc.generate(size=(32, 3)).to_torch()
    with pytest.raises(ValueError):
        pyhdc.stack([a, b])
