"""Tests for the global backend/device preferences (pyhdc.config)."""

import pytest

import pyhdc


@pytest.fixture(autouse=True)
def _reset_defaults():
    """Reset global defaults after every test to avoid cross-test leakage."""
    yield
    pyhdc.prefer_numpy()


def test_default_backend_is_numpy():
    assert pyhdc.get_default_backend() == "numpy"
    assert pyhdc.get_default_device() is None


def test_new_encoding_inherits_default_numpy():
    enc = pyhdc.MAP_C(dimension=64)
    assert enc.backend == "numpy"
    assert enc.generate().backend == "numpy"


def test_prefer_numpy_resets():
    pyhdc.prefer_cpu()
    pyhdc.prefer_numpy()
    assert pyhdc.get_default_backend() == "numpy"
    assert pyhdc.get_default_device() is None


def test_prefer_cpu_sets_device():
    pyhdc.prefer_cpu()
    assert pyhdc.get_default_device() == "cpu"


@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
def test_prefer_torch_changes_default():
    pyhdc.prefer_torch()
    assert pyhdc.get_default_backend() == "torch"
    enc = pyhdc.MAP_C(dimension=64)
    assert enc.backend == "torch"
    assert enc.generate().backend == "torch"


@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
def test_explicit_backend_overrides_torch_default():
    pyhdc.prefer_torch()
    enc = pyhdc.MAP_C(dimension=64, backend="numpy")
    assert enc.backend == "numpy"


@pytest.mark.skipif(pyhdc.TORCH_AVAILABLE, reason="requires torch to be absent")
def test_prefer_torch_raises_without_torch():
    with pytest.raises(ImportError):
        pyhdc.prefer_torch()


def test_prefer_cuda_errors_when_unavailable():
    if not pyhdc.TORCH_AVAILABLE:
        with pytest.raises(ImportError):
            pyhdc.prefer_cuda()
        return
    import torch

    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            pyhdc.prefer_cuda()
    else:
        pyhdc.prefer_cuda()
        assert pyhdc.get_default_backend() == "torch"
        assert str(pyhdc.get_default_device()).startswith("cuda")
