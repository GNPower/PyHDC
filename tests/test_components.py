"""Tests for pyhdc.components submodules."""

import numpy as np
import pytest

from pyhdc.components.binding.multiplication import ElementMultiplication
from pyhdc.components.binding.xor import ExclusiveOr
from pyhdc.components.bundling.addition import ElementAddition
from pyhdc.components.bundling.binary import Disjunction, DisjunctionThinned
from pyhdc.components.elements.bernoulli import BernoulliBinary, BernoulliBipolar
from pyhdc.components.similarity.angle import AngleDistance
from pyhdc.components.similarity.cosine import CosineSimilarity
from pyhdc.components.similarity.hamming import HammingDistance
from pyhdc.components.similarity.overlap import Overlap
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

    def test_2d_batch_returns_array(self):
        a = np.random.randn(DIM, 4).astype(np.float32)
        b = np.random.randn(DIM, 4).astype(np.float32)
        sims = CosineSimilarity(a, b)
        assert sims.shape == (4,)
        assert np.all(sims >= -1.0) and np.all(sims <= 1.0)

    def test_2d_batch_identical_rows_is_one(self):
        a = np.random.randn(DIM, 4).astype(np.float32)
        sims = CosineSimilarity(a, a)
        np.testing.assert_allclose(sims, np.ones(4), atol=1e-5)

    def test_single_2d_array_batched(self):
        rows = np.random.randn(DIM, 5).astype(np.float32)
        sims = CosineSimilarity(rows)
        assert sims.shape == (4,)  # sim(row_0, row_i) for i in 1..4
        assert np.all(sims >= -1.0) and np.all(sims <= 1.0)


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

    def test_2d_batch_returns_array(self):
        a = np.random.randint(0, 2, (DIM, 4), dtype=np.int32)
        b = np.random.randint(0, 2, (DIM, 4), dtype=np.int32)
        sims = HammingDistance(a, b)
        assert sims.shape == (4,)
        assert np.all(sims >= -1.0) and np.all(sims <= 1.0)

    def test_2d_batch_identical_rows_is_one(self):
        a = np.random.randint(0, 2, (DIM, 4), dtype=np.int32)
        sims = HammingDistance(a, a)
        np.testing.assert_allclose(sims, np.ones(4), atol=1e-10)

    def test_single_2d_array_batched(self):
        rows = np.random.randint(0, 2, (DIM, 5), dtype=np.int32)
        sims = HammingDistance(rows)
        assert sims.shape == (4,)
        assert np.all(sims >= -1.0) and np.all(sims <= 1.0)


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
        result = BernoulliBipolar(DIM, np.int32)
        unique = set(np.unique(result))
        assert unique.issubset({-1, 1})

    def test_bernoulli_bipolar_correct_length(self):
        result = BernoulliBipolar(DIM, np.int32)
        assert len(result) == DIM

    def test_bernoulli_bipolar_approximately_balanced(self):
        result = BernoulliBipolar(10000, np.int32)
        fraction_pos = np.mean(result == 1)
        assert 0.4 < fraction_pos < 0.6


class TestOverlap:
    def test_identical_sparse_vectors_is_one(self):
        v = np.zeros(DIM, dtype=np.int32)
        v[: DIM // 4] = 1
        sim = float(Overlap(v, v))
        assert abs(sim - 1.0) < 1e-10

    def test_no_overlap_is_minus_one(self):
        a = np.zeros(DIM, dtype=np.int32)
        b = np.zeros(DIM, dtype=np.int32)
        a[: DIM // 2] = 1
        b[DIM // 2 :] = 1
        sim = float(Overlap(a, b))
        assert abs(sim - (-1.0)) < 1e-10

    def test_range_is_minus_one_to_one(self):
        v1 = np.random.randint(0, 2, DIM, dtype=np.int32)
        v2 = np.random.randint(0, 2, DIM, dtype=np.int32)
        sim = float(Overlap(v1, v2))
        assert -1.0 <= sim <= 1.0

    def test_2d_batch_returns_array(self):
        a = np.random.randint(0, 2, (DIM, 4), dtype=np.int32)
        b = np.random.randint(0, 2, (DIM, 4), dtype=np.int32)
        sims = Overlap(a, b)
        assert sims.shape == (4,)

    def test_single_2d_array_batched(self):
        rows = np.random.randint(0, 2, (DIM, 5), dtype=np.int32)
        sims = Overlap(rows)
        assert sims.shape == (4,)


class TestAngleDistance:
    def test_identical_vectors_is_one(self):
        v = np.random.uniform(0, 2 * np.pi, DIM).astype(np.float32)
        sim = float(AngleDistance(v, v))
        assert abs(sim - 1.0) < 1e-5

    def test_range_is_minus_one_to_one(self):
        v1 = np.random.uniform(0, 2 * np.pi, DIM).astype(np.float32)
        v2 = np.random.uniform(0, 2 * np.pi, DIM).astype(np.float32)
        sim = float(AngleDistance(v1, v2))
        assert -1.0 <= sim <= 1.0

    def test_2d_batch_returns_array(self):
        a = np.random.uniform(0, 2 * np.pi, (DIM, 4)).astype(np.float32)
        b = np.random.uniform(0, 2 * np.pi, (DIM, 4)).astype(np.float32)
        sims = AngleDistance(a, b)
        assert sims.shape == (4,)

    def test_single_2d_array_batched(self):
        rows = np.random.uniform(0, 2 * np.pi, (DIM, 5)).astype(np.float32)
        sims = AngleDistance(rows)
        assert sims.shape == (4,)


class TestDisjunction:
    def test_or_of_two_binary_vectors(self):
        v1 = np.array([1, 0, 1, 0], dtype=np.int32)
        v2 = np.array([0, 1, 0, 0], dtype=np.int32)
        result = Disjunction(v1, v2)
        np.testing.assert_array_equal(result, [1, 1, 1, 0])

    def test_output_is_binary(self):
        v1 = np.random.randint(0, 2, DIM, dtype=np.int32)
        v2 = np.random.randint(0, 2, DIM, dtype=np.int32)
        result = Disjunction(v1, v2)
        assert set(np.unique(result)).issubset({0, 1})

    def test_output_shape(self):
        v1 = np.random.randint(0, 2, DIM, dtype=np.int32)
        v2 = np.random.randint(0, 2, DIM, dtype=np.int32)
        result = Disjunction(v1, v2)
        assert result.shape == (DIM,)

    def test_disjunction_superset_of_inputs(self):
        v1 = np.array([1, 0, 0, 0], dtype=np.int32)
        v2 = np.array([0, 1, 0, 0], dtype=np.int32)
        result = Disjunction(v1, v2)
        assert result[0] == 1 and result[1] == 1


class TestDisjunctionThinned:
    def test_output_is_binary(self):
        v1 = np.random.randint(0, 2, DIM, dtype=np.int32)
        v2 = np.random.randint(0, 2, DIM, dtype=np.int32)
        result = DisjunctionThinned(v1, v2)
        assert set(np.unique(result)).issubset({0, 1})

    def test_output_shape(self):
        v1 = np.random.randint(0, 2, DIM, dtype=np.int32)
        v2 = np.random.randint(0, 2, DIM, dtype=np.int32)
        result = DisjunctionThinned(v1, v2)
        assert result.shape == (DIM,)

    def test_density_respected(self):
        v1 = np.ones(DIM, dtype=np.int32)
        v2 = np.ones(DIM, dtype=np.int32)
        result = DisjunctionThinned(v1, v2, density=0.25)
        assert result.sum() <= int(np.ceil(DIM * 0.25))

    def test_no_thinning_when_already_sparse(self):
        v1 = np.zeros(DIM, dtype=np.int32)
        v1[:2] = 1
        v2 = np.zeros(DIM, dtype=np.int32)
        v2[2:4] = 1
        result = DisjunctionThinned(v1, v2, density=0.5)
        assert result.sum() == 4

    def test_result_subset_of_bundled(self):
        v1 = np.random.randint(0, 2, DIM, dtype=np.int32)
        v2 = np.random.randint(0, 2, DIM, dtype=np.int32)
        bundled = Disjunction(v1, v2)
        thinned = DisjunctionThinned(v1, v2, density=0.3)
        assert np.all(thinned[bundled == 0] == 0)


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


class TestBatchedReduction:
    def test_element_addition_batch_collapses(self):
        # (D=3, N=2): columns are the vectors to bundle
        batch = np.array([[1.0, 2.0], [3.0, 4.0], [-5.0, -6.0]], dtype=np.float32)
        result, _ = ElementAddition(batch)
        np.testing.assert_array_almost_equal(result, batch.sum(axis=1))

    def test_element_addition_batch_matches_args(self):
        batch = np.array([[1.0, 2.0], [3.0, 4.0], [-5.0, -6.0]], dtype=np.float32)
        from_batch, _ = ElementAddition(batch)
        from_args, _ = ElementAddition(batch[:, 0], batch[:, 1])
        np.testing.assert_array_almost_equal(from_batch, from_args)

    def test_disjunction_batch_collapses(self):
        batch = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.int32)
        result = Disjunction(batch)
        np.testing.assert_array_equal(result, [1, 1, 1, 0])


class TestSimilarityModes:
    def test_two_singles_returns_float(self):
        v = np.random.randn(DIM).astype(np.float32)
        sim = CosineSimilarity(v, v)
        assert isinstance(sim, float)
        assert abs(sim - 1.0) < 1e-5

    def test_vector_vs_batch_broadcast(self):
        v = np.random.randn(DIM).astype(np.float32)
        b = np.random.randn(DIM, 4).astype(np.float32)
        b[:, 0] = v
        sims = CosineSimilarity(v, b)
        assert sims.shape == (4,)
        assert abs(sims[0] - 1.0) < 1e-5


torch = pytest.importorskip("torch", reason="PyTorch not installed")


class TestSimilarityTorch:
    """Covers the PyTorch execution paths in all four similarity functions."""

    def test_cosine_1d_torch(self):
        v = torch.randn(DIM)
        sim = CosineSimilarity(v, v)
        assert abs(float(sim) - 1.0) < 1e-4

    def test_cosine_2d_torch(self):
        a = torch.randn(DIM, 4)
        b = torch.randn(DIM, 4)
        sims = CosineSimilarity(a, b)
        assert sims.shape == (4,)
        assert (sims >= -1.0).all() and (sims <= 1.0).all()

    def test_cosine_single_2d_torch(self):
        rows = torch.randn(DIM, 5)
        sims = CosineSimilarity(rows)
        assert sims.shape == (4,)

    def test_hamming_1d_torch(self):
        v = torch.randint(0, 2, (DIM,))
        sim = HammingDistance(v, v)
        assert abs(float(sim) - 1.0) < 1e-6

    def test_hamming_2d_torch(self):
        a = torch.randint(0, 2, (DIM, 4))
        b = torch.randint(0, 2, (DIM, 4))
        sims = HammingDistance(a, b)
        assert sims.shape == (4,)

    def test_hamming_single_2d_torch(self):
        rows = torch.randint(0, 2, (DIM, 5))
        sims = HammingDistance(rows)
        assert sims.shape == (4,)

    def test_overlap_1d_torch(self):
        v = torch.zeros(DIM, dtype=torch.int32)
        v[: DIM // 4] = 1
        sim = Overlap(v, v)
        assert abs(float(sim) - 1.0) < 1e-6

    def test_overlap_2d_torch(self):
        a = torch.randint(0, 2, (DIM, 4))
        b = torch.randint(0, 2, (DIM, 4))
        sims = Overlap(a, b)
        assert sims.shape == (4,)

    def test_overlap_single_2d_torch(self):
        rows = torch.randint(0, 2, (DIM, 5))
        sims = Overlap(rows)
        assert sims.shape == (4,)

    def test_angle_1d_torch(self):
        v = torch.rand(DIM) * 2 * 3.14159
        sim = AngleDistance(v, v)
        assert abs(float(sim) - 1.0) < 1e-4

    def test_angle_2d_torch(self):
        a = torch.rand(DIM, 4) * 2 * 3.14159
        b = torch.rand(DIM, 4) * 2 * 3.14159
        sims = AngleDistance(a, b)
        assert sims.shape == (4,)

    def test_angle_single_2d_torch(self):
        rows = torch.rand(DIM, 5) * 2 * 3.14159
        sims = AngleDistance(rows)
        assert sims.shape == (4,)

    def test_disjunction_thinned_torch(self):
        v1 = torch.ones(DIM, dtype=torch.int8)
        v2 = torch.ones(DIM, dtype=torch.int8)
        result = DisjunctionThinned(v1, v2, density=0.25)
        assert result.sum() <= int(DIM * 0.25) + 1

    def test_disjunction_thinned_torch_sparse_passthrough(self):
        v1 = torch.zeros(DIM, dtype=torch.int8)
        v1[:2] = 1
        v2 = torch.zeros(DIM, dtype=torch.int8)
        v2[2:4] = 1
        result = DisjunctionThinned(v1, v2, density=0.5)
        assert result.sum() == 4
