"""Tests for pyhdc.components submodules."""

import numpy as np

from pyhdc.components.binding.multiplication import ElementMultiplication
from pyhdc.components.binding.xor import ExclusiveOr
from pyhdc.components.bundling.addition import ElementAddition
from pyhdc.components.elements.bernoulli import BernoulliBinary, BernoulliBiploar
from pyhdc.components.similarity.cosine import CosineSimilarity
from pyhdc.components.similarity.hamming import HammingDistance
from pyhdc.components.thinning import NoThin

DIM = 256


class TestCosineSimilarity:
    def test_identical_vectors_is_one(self):
        v = np.random.rand(DIM).astype(np.float32)
        sim = float(CosineSimilarity(v, v))
        assert abs(sim - 1.0) < 1e-5

    def test_opposite_vectors_is_minus_one(self):
        v = np.ones(DIM, dtype=np.float32)
        sim = float(CosineSimilarity(v, -v))
        assert abs(sim + 1.0) < 1e-5

    def test_orthogonal_vectors_is_zero(self):
        v1 = np.zeros(DIM, dtype=np.float32)
        v2 = np.zeros(DIM, dtype=np.float32)
        v1[0] = 1.0
        v2[1] = 1.0
        sim = float(CosineSimilarity(v1, v2))
        assert abs(sim) < 1e-5

    def test_range_is_minus_one_to_one(self):
        v1 = np.random.randn(DIM).astype(np.float32)
        v2 = np.random.randn(DIM).astype(np.float32)
        sim = float(CosineSimilarity(v1, v2))
        assert -1.0 <= sim <= 1.0

    def test_returns_scalar(self):
        v1 = np.random.rand(DIM).astype(np.float32)
        v2 = np.random.rand(DIM).astype(np.float32)
        sim = CosineSimilarity(v1, v2)
        assert np.isscalar(sim) or sim.ndim == 0


class TestHammingDistance:
    def test_identical_vectors_is_one(self):
        v = np.random.randint(0, 2, DIM, dtype=np.int32)
        sim = float(HammingDistance(v, v))
        assert abs(sim - 1.0) < 1e-10

    def test_complementary_vectors_is_minus_one(self):
        v = np.zeros(DIM, dtype=np.int32)
        v_comp = np.ones(DIM, dtype=np.int32)
        sim = float(HammingDistance(v, v_comp))
        assert abs(sim - (-1.0)) < 1e-10

    def test_range_is_minus_one_to_one(self):
        v1 = np.random.randint(0, 2, DIM, dtype=np.int32)
        v2 = np.random.randint(0, 2, DIM, dtype=np.int32)
        sim = float(HammingDistance(v1, v2))
        assert -1.0 <= sim <= 1.0

    def test_approximately_zero_for_random_pair(self):
        np.random.seed(42)
        v1 = np.random.randint(0, 2, 10000, dtype=np.int32)
        v2 = np.random.randint(0, 2, 10000, dtype=np.int32)
        sim = float(HammingDistance(v1, v2))
        assert -0.2 < sim < 0.2


class TestElementAddition:
    def test_sum_of_two_vectors(self):
        # Use non-zero-sum elements to avoid the random zone randomization
        v1 = np.array([1.0, 2.0, -1.0, -2.0], dtype=np.float32)
        v2 = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
        result, meta = ElementAddition(v1, v2)
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, -2.0, -3.0])

    def test_returns_metadata_dict(self):
        v1 = np.ones(DIM, dtype=np.float32)
        v2 = np.ones(DIM, dtype=np.float32)
        result, meta = ElementAddition(v1, v2)
        assert isinstance(meta, dict)

    def test_metadata_has_random_zone_count(self):
        v1 = np.array([1.0, -1.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 1.0, 0.0], dtype=np.float32)
        result, meta = ElementAddition(v1, v2)
        assert "random_zone_count" in meta

    def test_result_shape_matches_input(self):
        v1 = np.random.randn(DIM).astype(np.float32)
        v2 = np.random.randn(DIM).astype(np.float32)
        result, _ = ElementAddition(v1, v2)
        assert result.shape == (DIM,)

    def test_single_vector_passthrough(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result, _ = ElementAddition(v)
        np.testing.assert_array_almost_equal(result, v)


class TestExclusiveOr:
    def test_xor_is_self_inverse(self):
        v = np.random.randint(0, 2, DIM, dtype=np.int32)
        w = np.random.randint(0, 2, DIM, dtype=np.int32)
        bound = ExclusiveOr(v, w)
        recovered = ExclusiveOr(bound, w)
        np.testing.assert_array_equal(recovered, v)

    def test_xor_same_vector_is_zero(self):
        v = np.random.randint(0, 2, DIM, dtype=np.int32)
        result = ExclusiveOr(v, v)
        assert np.all(result == 0)

    def test_xor_output_shape(self):
        v1 = np.random.randint(0, 2, DIM, dtype=np.int32)
        v2 = np.random.randint(0, 2, DIM, dtype=np.int32)
        result = ExclusiveOr(v1, v2)
        assert result.shape == (DIM,)

    def test_xor_output_is_binary(self):
        v1 = np.random.randint(0, 2, DIM, dtype=np.int32)
        v2 = np.random.randint(0, 2, DIM, dtype=np.int32)
        result = ExclusiveOr(v1, v2)
        unique = set(np.unique(result))
        assert unique.issubset({0, 1})


class TestElementMultiplication:
    def test_multiply_bipolar_is_self_inverse(self):
        v = np.random.choice([-1, 1], DIM).astype(np.float32)
        w = np.random.choice([-1, 1], DIM).astype(np.float32)
        bound = ElementMultiplication(v, w)
        recovered = ElementMultiplication(bound, w)
        np.testing.assert_array_almost_equal(recovered, v)

    def test_multiply_self_is_ones(self):
        v = np.random.choice([-1, 1], DIM).astype(np.float32)
        result = ElementMultiplication(v, v)
        np.testing.assert_array_almost_equal(result, np.ones(DIM, dtype=np.float32))

    def test_output_shape(self):
        v1 = np.random.choice([-1, 1], DIM).astype(np.float32)
        v2 = np.random.choice([-1, 1], DIM).astype(np.float32)
        result = ElementMultiplication(v1, v2)
        assert result.shape == (DIM,)


class TestBernoulliElements:
    def test_bernoulli_binary_values_in_zero_one(self):
        result = BernoulliBinary(DIM, np.int32)
        unique = set(np.unique(result))
        assert unique.issubset({0, 1})

    def test_bernoulli_binary_correct_length(self):
        result = BernoulliBinary(DIM, np.int32)
        assert len(result) == DIM

    def test_bernoulli_binary_approximately_half(self):
        result = BernoulliBinary(10000, np.int32)
        fraction = np.mean(result)
        assert 0.4 < fraction < 0.6

    def test_bernoulli_bipolar_values_in_minus_one_one(self):
        result = BernoulliBiploar(DIM, np.int32)
        unique = set(np.unique(result))
        assert unique.issubset({-1, 1})

    def test_bernoulli_bipolar_correct_length(self):
        result = BernoulliBiploar(DIM, np.int32)
        assert len(result) == DIM

    def test_bernoulli_bipolar_approximately_balanced(self):
        result = BernoulliBiploar(10000, np.int32)
        fraction_pos = np.mean(result == 1)
        assert 0.4 < fraction_pos < 0.6


class TestNoThin:
    def test_identity_on_float_array(self):
        v = np.random.rand(DIM).astype(np.float32)
        result = NoThin(v)
        np.testing.assert_array_equal(result, v)

    def test_identity_on_binary_array(self):
        v = np.random.randint(0, 2, DIM, dtype=np.int32)
        result = NoThin(v)
        np.testing.assert_array_equal(result, v)

    def test_returns_same_array(self):
        v = np.array([1.0, 2.0, 3.0])
        result = NoThin(v)
        assert result is v
