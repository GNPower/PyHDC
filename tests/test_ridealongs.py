"""Tests for the C4-C6 ride-along components (pyhdc.components.ridealongs)."""

import numpy as np
import pytest

import pyhdc
from pyhdc.components import (
    hard_quantize,
    multibind,
    multibundle,
    multirandsel,
    multiset,
    randsel,
    soft_quantize,
)

D, N = 8, 5


def test_randsel_picks_from_inputs():
    np.random.seed(0)
    data = np.arange(D * N).reshape(D, N)
    out = randsel(data)
    assert out.shape == (D,)
    assert all(out[i] in data[i] for i in range(D))


def test_randsel_weighted():
    data = np.arange(D * N).reshape(D, N)
    out = randsel(data, p=[0, 0, 0, 0, 1.0])  # all weight on the last column
    assert np.array_equal(out, data[:, -1])


def test_randsel_weights_need_not_sum_to_one():
    # Weights are normalized internally, so un-normalized weights work (and match
    # the torch backend, which auto-normalizes via multinomial).
    data = np.arange(D * N).reshape(D, N)
    out = randsel(data, p=[0, 0, 0, 0, 5.0])  # sums to 5, all mass on last column
    assert np.array_equal(out, data[:, -1])


def test_multirandsel_shape():
    assert multirandsel(np.random.randn(D, N), 3).shape == (D, 3)


def test_multiset_is_sum():
    data = np.random.randn(D, N)
    assert np.allclose(multiset(data), data.sum(axis=-1))
    assert multibundle is multiset


def test_multibind_is_prod():
    data = np.random.choice([-1, 1], size=(D, N))
    assert np.array_equal(multibind(data), data.prod(axis=-1))


def test_quantize():
    x = np.array([-2.0, 0.0, 3.0])
    assert np.array_equal(hard_quantize(x), np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(soft_quantize(x), np.tanh(x))
    assert np.allclose(soft_quantize(x, temperature=2.0), np.tanh(x / 2.0))


@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
def test_ridealongs_torch():
    import torch

    data = torch.arange(D * N).reshape(D, N)
    assert tuple(randsel(data).shape) == (D,)
    assert tuple(multirandsel(data, 3).shape) == (D, 3)
    assert torch.equal(multiset(data), data.sum(dim=-1))
    bip = torch.tensor(np.random.choice([-1, 1], size=(D, N)))
    assert torch.equal(multibind(bip), bip.prod(dim=-1))
    assert torch.equal(
        hard_quantize(torch.tensor([-2.0, 0.0, 3.0])), torch.tensor([-1.0, 0.0, 1.0])
    )
