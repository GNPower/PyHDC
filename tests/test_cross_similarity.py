"""Tests for cross similarity (mode="cross")."""

import numpy as np
import pytest

import pyhdc
from pyhdc.components.input_formatting import _normalize_similarity
from pyhdc.components.similarity import HammingDistance

D, P, M = 16, 3, 5


def _pairwise_ref(enc, A, B):
    out = np.empty((A.shape[1], B.shape[1]))
    for p in range(A.shape[1]):
        for m in range(B.shape[1]):
            out[p, m] = float(enc.similarity(A[:, p], B[:, m]))
    return out


@pytest.mark.parametrize(
    "name,atol",
    [("MAP_C", 1e-6), ("BSC", 1e-9), ("BSDC_S", 1e-9), ("FHRR", 1e-6)],
)
def test_cross_matches_pairwise(name, atol):
    np.random.seed(0)
    enc = getattr(pyhdc, name)(dimension=D)
    A, B = enc.generate((D, P)), enc.generate((D, M))
    cross = np.asarray(enc.similarity(A, B, mode="cross"))
    assert cross.shape == (P, M)
    assert np.allclose(cross, _pairwise_ref(enc, A.data, B.data), atol=atol)


def test_no_dpm_materialization():
    enc = pyhdc.MAP_C(dimension=D)
    cross = np.asarray(
        enc.similarity(enc.generate((D, P)), enc.generate((D, M)), mode="cross")
    )
    assert cross.ndim == 2 and cross.nbytes == P * M * cross.itemsize


def test_p_not_equal_m():
    enc = pyhdc.MAP_C(dimension=D)
    cross = np.asarray(
        enc.similarity(enc.generate((D, 2)), enc.generate((D, 7)), mode="cross")
    )
    assert cross.shape == (2, 7)


def test_overlap_asymmetric():
    enc = pyhdc.BSDC_S(dimension=64)
    A, B = enc.generate((64, 4)), enc.generate((64, 6))
    ab = np.asarray(enc.similarity(A, B, mode="cross"))
    ba = np.asarray(enc.similarity(B, A, mode="cross"))
    assert not np.allclose(ab, ba.T)


def test_cosine_zero_norm_returns_zero_not_nan():
    enc = pyhdc.MAP_C(dimension=D)
    A = enc.generate((D, 2))
    bdata = enc.generate((D, 2)).data.copy()
    bdata[:, 0] = 0.0
    from pyhdc.components.similarity import CosineSimilarity

    cross = np.asarray(CosineSimilarity(A.data, bdata, mode="cross"))
    assert np.all(np.isfinite(cross))
    assert np.allclose(cross[:, 0], 0.0)


def test_binary_float_cast_exact():
    np.random.seed(1)
    A = np.random.randint(0, 2, (D, P)).astype(np.int8)
    B = np.random.randint(0, 2, (D, M)).astype(np.int8)
    cross = HammingDistance(A, B, mode="cross")
    assert cross.dtype == np.float64
    ref = np.empty((P, M))
    for p in range(P):
        for m in range(M):
            ref[p, m] = float(HammingDistance(A[:, p], B[:, m]))
    assert np.allclose(cross, ref, atol=1e-12)


def test_gate_raises_for_unknown_metric():
    enc = pyhdc.MAP_C(dimension=D)
    enc._spec.similarity_fn = lambda *a, **k: 0
    with pytest.raises(NotImplementedError):
        enc.similarity(enc.generate((D, 2)), enc.generate((D, 2)), mode="cross")


def test_axis_and_cross_mutually_exclusive():
    enc = pyhdc.MAP_C(dimension=D)
    with pytest.raises(ValueError):
        enc.similarity(enc.generate((D, 2)), enc.generate((D, 2)), axis=1, mode="cross")


def test_cross_requires_two_batches():
    enc = pyhdc.MAP_C(dimension=D)
    with pytest.raises(ValueError):
        enc.similarity(enc.generate((D, 2)), mode="cross")


def test_normalize_similarity_cross_validation():
    with pytest.raises(ValueError):
        _normalize_similarity(np.zeros((D, 2)), mode="cross")  # one input
    with pytest.raises(ValueError):
        _normalize_similarity(
            np.zeros((D, 2)), np.zeros((D + 1, 2)), mode="cross"
        )  # D mismatch
    with pytest.raises(ValueError):
        _normalize_similarity(np.zeros(D), np.zeros(D), mode="cross")  # 1D inputs


def test_module_level_and_hypervector_cross_agree():
    enc = pyhdc.MAP_C(dimension=D)
    A, B = enc.generate((D, 3)), enc.generate((D, 4))
    assert np.allclose(
        np.asarray(A.similarity(B, mode="cross")),
        np.asarray(pyhdc.similarity(A, B, mode="cross")),
    )


def test_pairwise_default_unchanged():
    enc = pyhdc.MAP_C(dimension=D)
    a, b = enc.generate(), enc.generate()
    assert isinstance(enc.similarity(a, b), float)
    assert np.asarray(enc.similarity(enc.generate((D, 5)))).shape == (4,)


@pytest.mark.skipif(not pyhdc.TORCH_AVAILABLE, reason="PyTorch not installed")
def test_cross_torch_parity():
    np.random.seed(2)
    enc = pyhdc.BSC(dimension=D, backend="torch")
    A, B = enc.generate((D, 3)), enc.generate((D, 5))
    ct = enc.similarity(A, B, mode="cross").cpu().numpy()
    cn = pyhdc.BSC(dimension=D)._spec.similarity_fn(
        A.data.cpu().numpy(), B.data.cpu().numpy(), mode="cross"
    )
    assert np.allclose(ct, cn, atol=1e-6)
