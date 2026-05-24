"""Tests for the Hypervector class and BackendManager."""

import numpy as np
import pytest

import pyhdc
from pyhdc.hypervector import BackendManager, Hypervector

DIM = 512


@pytest.fixture(scope="module")
def enc():
    return pyhdc.MAP_C(dimension=DIM)


@pytest.fixture
def hv(enc):
    return enc.generate()


class TestHypervectorProperties:
    def test_generate_returns_hypervector(self, hv):
        assert isinstance(hv, Hypervector)

    def test_shape(self, hv):
        assert hv.shape == (DIM,)

    def test_ndim(self, hv):
        assert hv.ndim == 1

    def test_backend_is_numpy(self, hv):
        assert hv.backend == "numpy"

    def test_dtype_is_float(self, hv):
        assert np.issubdtype(hv.dtype, np.floating)

    def test_device_is_none_for_numpy(self, hv):
        assert hv.device is None

    def test_repr_contains_class_info(self, hv):
        r = repr(hv)
        assert "Hypervector" in r
        assert "MAP_C" in r

    def test_len_equals_dimension(self, hv):
        assert len(hv) == DIM

    def test_data_is_array(self, hv):
        assert isinstance(hv.data, np.ndarray)

    def test_encoding_attribute(self, enc, hv):
        assert hv.encoding is enc


class TestHypervectorSlicing:
    def test_getitem_slice(self, hv):
        sliced = hv[0:10]
        assert isinstance(sliced, Hypervector)
        assert sliced.shape == (10,)

    def test_getitem_single_index_returns_hypervector(self, hv):
        elem = hv[0]
        assert isinstance(elem, Hypervector)

    def test_getitem_preserves_backend(self, hv):
        sliced = hv[0:10]
        assert sliced.backend == "numpy"


class TestHypervectorMetadata:
    def test_metadata_is_dict(self, hv):
        meta = hv.get_metadata()
        assert isinstance(meta, dict)

    def test_metadata_copy_not_reference(self, enc):
        hv = enc.generate()
        meta = hv.get_metadata()
        meta["injected"] = True
        assert "injected" not in hv.get_metadata()


class TestHypervectorBackendConversion:
    def test_to_numpy_idempotent(self, hv):
        hv2 = hv.to_numpy()
        assert hv2 is hv

    def test_to_numpy_from_numpy_returns_same(self, hv):
        result = hv.to_numpy()
        assert result.backend == "numpy"
        np.testing.assert_array_equal(result.data, hv.data)


class TestSimilarity:
    def test_self_similarity_is_one(self, enc, hv):
        sim = hv.similarity(hv)
        assert abs(float(sim) - 1.0) < 1e-4

    def test_similarity_range(self, enc):
        hv1 = enc.generate()
        hv2 = enc.generate()
        sim = float(hv1.similarity(hv2))
        assert -1.0 <= sim <= 1.0

    def test_similarity_is_symmetric(self, enc):
        hv1 = enc.generate()
        hv2 = enc.generate()
        sim_ab = float(hv1.similarity(hv2))
        sim_ba = float(hv2.similarity(hv1))
        assert abs(sim_ab - sim_ba) < 1e-5

    def test_random_pair_not_self_similar(self, enc):
        hv1 = enc.generate()
        hv2 = enc.generate()
        sim = abs(float(hv1.similarity(hv2)))
        assert sim < 0.5


class TestBundle:
    def test_bundle_returns_hypervector(self, enc):
        hv1 = enc.generate()
        hv2 = enc.generate()
        result = hv1.bundle(hv2)
        assert isinstance(result, Hypervector)

    def test_bundle_preserves_dimension(self, enc):
        hv1 = enc.generate()
        hv2 = enc.generate()
        result = hv1.bundle(hv2)
        assert result.shape == (DIM,)

    def test_bundle_three_vectors(self, enc):
        hvs = [enc.generate() for _ in range(3)]
        result = hvs[0].bundle(hvs[1], hvs[2])
        assert isinstance(result, Hypervector)
        assert result.shape == (DIM,)

    def test_bundle_similar_to_inputs(self, enc):
        hv1 = enc.generate()
        hv2 = enc.generate()
        bundled = enc.bundle(hv1, hv2)
        sim1 = float(enc.similarity(bundled, hv1))
        sim_rand = float(enc.similarity(enc.generate(), hv1))
        assert sim1 > sim_rand


class TestBind:
    def test_bind_returns_hypervector(self, enc):
        hv1 = enc.generate()
        hv2 = enc.generate()
        result = hv1.bind(hv2)
        assert isinstance(result, Hypervector)

    def test_bind_preserves_dimension(self, enc):
        hv1 = enc.generate()
        hv2 = enc.generate()
        result = hv1.bind(hv2)
        assert result.shape == (DIM,)

    def test_bound_dissimilar_to_inputs(self, enc):
        hv1 = enc.generate()
        hv2 = enc.generate()
        bound = enc.bind(hv1, hv2)
        sim_with_hv1 = abs(float(enc.similarity(bound, hv1)))
        assert sim_with_hv1 < 0.5


class TestBackendManager:
    def test_get_backend_numpy(self):
        arr = np.zeros(10)
        assert BackendManager.get_backend(arr) == "numpy"

    def test_to_numpy_from_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = BackendManager.to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_get_device_numpy_is_none(self):
        arr = np.zeros(10)
        assert BackendManager.get_device(arr) is None
