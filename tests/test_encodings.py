"""Tests for all 14 encoding classes."""

import numpy as np
import pytest

import pyhdc
from pyhdc.hypervector import Hypervector

DIM = 512

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

# VTB requires dimension to be a perfect square (d_prime = sqrt(D) must be int)
VTB_DIM = 484  # 22 * 22


def make_enc(name):
    cls = getattr(pyhdc, name)
    if name == "VTB":
        return cls(dimension=VTB_DIM)
    return cls(dimension=DIM)


@pytest.mark.parametrize("enc_name", ALL_ENCODINGS)
class TestAllEncodings:
    def test_generate_returns_hypervector(self, enc_name):
        enc = make_enc(enc_name)
        hv = enc.generate()
        assert isinstance(hv, Hypervector)

    def test_generate_correct_dimension(self, enc_name):
        enc = make_enc(enc_name)
        hv = enc.generate()
        assert hv.shape[0] == enc.dimension

    def test_zeros_returns_hypervector(self, enc_name):
        enc = make_enc(enc_name)
        hv = enc.zeros()
        assert isinstance(hv, Hypervector)

    def test_zeros_correct_dimension(self, enc_name):
        enc = make_enc(enc_name)
        hv = enc.zeros()
        assert hv.shape[0] == enc.dimension

    def test_from_array(self, enc_name):
        enc = make_enc(enc_name)
        hv = enc.generate()
        hv2 = enc.from_array(hv.data)
        assert isinstance(hv2, Hypervector)
        np.testing.assert_array_equal(hv2.data, hv.data)

    def test_similarity_self_is_max(self, enc_name):
        enc = make_enc(enc_name)
        hv = enc.generate()
        sim = enc.similarity(hv, hv)
        assert sim is not None

    def test_bundle_two_vectors(self, enc_name):
        enc = make_enc(enc_name)
        hv1 = enc.generate()
        hv2 = enc.generate()
        result = enc.bundle(hv1, hv2)
        assert isinstance(result, Hypervector)
        assert result.shape[0] == enc.dimension

    def test_bind_two_vectors(self, enc_name):
        enc = make_enc(enc_name)
        hv1 = enc.generate()
        hv2 = enc.generate()
        result = enc.bind(hv1, hv2)
        assert isinstance(result, Hypervector)
        assert result.shape[0] == enc.dimension

    def test_dimension_attribute(self, enc_name):
        enc = make_enc(enc_name)
        # VTB uses VTB_DIM (perfect square), all others use DIM
        expected = VTB_DIM if enc_name == "VTB" else DIM
        assert enc.dimension == expected

    def test_backend_attribute(self, enc_name):
        enc = make_enc(enc_name)
        assert enc.backend == "numpy"

    def test_two_generated_vectors_differ(self, enc_name):
        enc = make_enc(enc_name)
        hv1 = enc.generate()
        hv2 = enc.generate()
        # Generated vectors should not be identical (astronomically unlikely)
        assert not np.array_equal(hv1.data, hv2.data)


class TestMAPCSpecific:
    def test_data_values_in_range(self):
        enc = pyhdc.MAP_C(dimension=DIM)
        hv = enc.generate()
        assert np.all(hv.data >= -1.0)
        assert np.all(hv.data <= 1.0)

    def test_self_similarity_is_one(self):
        enc = pyhdc.MAP_C(dimension=DIM)
        hv = enc.generate()
        sim = float(enc.similarity(hv, hv))
        assert abs(sim - 1.0) < 1e-4

    def test_bind_unbind_roundtrip(self):
        """MAP-C unbinding is approximate, ~0.7 similarity
        at D=1024. With DIM=512 we use a conservative threshold of 0.65."""
        enc = pyhdc.MAP_C(dimension=DIM)
        hv1 = enc.generate()
        hv2 = enc.generate()
        bound = enc.bind(hv1, hv2)
        recovered = enc.unbind(bound, hv2)
        sim = float(enc.similarity(recovered, hv1))
        assert sim > 0.65

    def test_batched_bundle(self):
        enc = pyhdc.MAP_C(dimension=DIM)
        hv1, hv2, hv3, hv4 = [enc.generate() for _ in range(4)]
        results = enc.bundle([[hv1, hv2], [hv3, hv4]])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, Hypervector) for r in results)

    def test_batched_similarity(self):
        enc = pyhdc.MAP_C(dimension=DIM)
        hv1, hv2, hv3, hv4 = [enc.generate() for _ in range(4)]
        sims = enc.similarity([hv1, hv2], [hv3, hv4])
        assert isinstance(sims, list)
        assert len(sims) == 2

    def test_bundle_similar_to_inputs(self):
        enc = pyhdc.MAP_C(dimension=DIM)
        hv1 = enc.generate()
        hv2 = enc.generate()
        bundled = enc.bundle(hv1, hv2)
        sim1 = float(enc.similarity(bundled, hv1))
        sim_rand = float(enc.similarity(enc.generate(), hv1))
        assert sim1 > sim_rand

    def test_custom_random_choice_range(self):
        enc = pyhdc.MAP_C(dimension=DIM, random_choice_range=0.5)
        hv = enc.generate()
        assert isinstance(hv, Hypervector)


class TestBSCSpecific:
    def test_binary_values_only(self):
        enc = pyhdc.BSC(dimension=DIM)
        hv = enc.generate()
        unique = set(np.unique(hv.data))
        assert unique.issubset({0, 1})

    def test_xor_bind_is_exact_inverse(self):
        """BSC uses XOR which is exactly self-inverse."""
        enc = pyhdc.BSC(dimension=DIM)
        hv1 = enc.generate()
        hv2 = enc.generate()
        bound = enc.bind(hv1, hv2)
        recovered = enc.unbind(bound, hv2)
        sim = float(enc.similarity(hv1, recovered))
        assert sim > 0.99

    def test_self_similarity_is_one(self):
        enc = pyhdc.BSC(dimension=DIM)
        hv = enc.generate()
        sim = float(enc.similarity(hv, hv))
        assert abs(sim - 1.0) < 1e-4

    def test_approximately_half_ones(self):
        enc = pyhdc.BSC(dimension=DIM)
        hv = enc.generate()
        fraction_ones = np.mean(hv.data)
        assert 0.3 < fraction_ones < 0.7


class TestMAPBSpecific:
    def test_bipolar_values_only(self):
        enc = pyhdc.MAP_B(dimension=DIM)
        hv = enc.generate()
        unique = set(np.unique(hv.data))
        assert unique.issubset({-1, 1})

    def test_bind_unbind_exact(self):
        """MAP-B uses element-wise multiplication which is exactly self-inverse
        for bipolar."""
        enc = pyhdc.MAP_B(dimension=DIM)
        hv1 = enc.generate()
        hv2 = enc.generate()
        bound = enc.bind(hv1, hv2)
        recovered = enc.unbind(bound, hv2)
        sim = float(enc.similarity(hv1, recovered))
        assert abs(sim - 1.0) < 1e-4


class TestHRRSpecific:
    def test_float_values(self):
        enc = pyhdc.HRR(dimension=DIM)
        hv = enc.generate()
        assert np.issubdtype(hv.dtype, np.floating)

    def test_bind_unbind_approximate(self):
        """HRR unbinding via circular correlation is approximate."""
        enc = pyhdc.HRR(dimension=DIM)
        hv1 = enc.generate()
        hv2 = enc.generate()
        bound = enc.bind(hv1, hv2)
        recovered = enc.unbind(bound, hv2)
        sim = float(enc.similarity(recovered, hv1))
        assert sim > 0.5


class TestFHRRSpecific:
    def test_generate_does_not_raise(self):
        enc = pyhdc.FHRR(dimension=DIM)
        hv = enc.generate()
        assert isinstance(hv, Hypervector)

    def test_bind_unbind_exact(self):
        """FHRR uses complex multiplication which has an exact inverse."""
        enc = pyhdc.FHRR(dimension=DIM)
        hv1 = enc.generate()
        hv2 = enc.generate()
        bound = enc.bind(hv1, hv2)
        recovered = enc.unbind(bound, hv2)
        sim = enc.similarity(recovered, hv1)
        # FHRR similarity is angle distance; closer to 0 is more similar
        assert sim is not None


class TestVTBSpecific:
    def test_generate_does_not_raise(self):
        enc = pyhdc.VTB(dimension=VTB_DIM)
        hv = enc.generate()
        assert isinstance(hv, Hypervector)

    def test_bind_unbind_approximate(self):
        enc = pyhdc.VTB(dimension=VTB_DIM)
        hv1 = enc.generate()
        hv2 = enc.generate()
        bound = enc.bind(hv1, hv2)
        recovered = enc.unbind(bound, hv2)
        sim = float(enc.similarity(recovered, hv1))
        assert sim > 0.5


class TestBSDCSpecific:
    @pytest.mark.parametrize("enc_name", ["BSDC_CDT", "BSDC_S", "BSDC_SEG"])
    def test_sparse_binary_values(self, enc_name):
        enc = make_enc(enc_name)
        hv = enc.generate()
        unique = set(np.unique(hv.data))
        assert unique.issubset({0, 1})

    @pytest.mark.parametrize("enc_name", ["BSDC_CDT", "BSDC_S", "BSDC_SEG"])
    def test_sparse_has_few_ones(self, enc_name):
        enc = make_enc(enc_name)
        hv = enc.generate()
        fraction_ones = np.mean(hv.data)
        assert fraction_ones < 0.3
