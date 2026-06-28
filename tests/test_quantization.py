"""Tests for the quantization components (hard_quantize, soft_quantize)."""

import numpy as np
import pytest

import pyhdc
from pyhdc.components.quantization import hard_quantize, soft_quantize


def test_hard_quantize_is_sign():
    x = np.array([-2.0, 0.0, 3.0])
    assert np.array_equal(hard_quantize(x), np.array([-1.0, 0.0, 1.0]))


def test_soft_quantize_is_tanh():
    x = np.array([-2.0, 0.0, 3.0])
    assert np.allclose(soft_quantize(x), np.tanh(x))
    assert np.allclose(soft_quantize(x, temperature=2.0), np.tanh(x / 2.0))


@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
def test_quantize_torch():
    import torch

    assert torch.equal(
        hard_quantize(torch.tensor([-2.0, 0.0, 3.0])), torch.tensor([-1.0, 0.0, 1.0])
    )
    x = torch.tensor([-2.0, 0.0, 3.0])
    assert torch.allclose(soft_quantize(x), torch.tanh(x))
