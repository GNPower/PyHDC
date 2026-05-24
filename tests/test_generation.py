"""Tests for all HDCGenerator implementations."""
import numpy as np
import pytest

from pyhdc.generation.base import DefaultGenerator
from pyhdc.generation.dlfsr import FibonacciDLFSRGenerator
from pyhdc.generation.lca import ElementaryLCAGenerator
from pyhdc.generation.lcg import (
    CommonLCGGenerators,
    LCGGenerator,
    MultiplicativeLCGGenerator,
)
from pyhdc.generation.lfsr import (
    CommonLFSRGenerators,
    FibonacciLFSRGenerator,
    GaloisLFSRGenerator,
    LFSRGenerator,
)
from pyhdc.generation.pcg import CommonPCGGenerators, PCGGenerator
from pyhdc.generation.shifted_counter import (
    CommonCounterGenerators,
    FeistelCounterGenerator,
)
from pyhdc.generation.xorshift import (
    CommonXorshiftGenerators,
    Xorshift32Generator,
    Xorshift64Generator,
)

LENGTH = 100


class TestDefaultGenerator:
    def test_generate_bits_correct_length(self):
        gen = DefaultGenerator()
        bits = gen.generate_bits(LENGTH)
        assert len(bits) == LENGTH

    def test_generate_bits_binary_values(self):
        gen = DefaultGenerator()
        bits = gen.generate_bits(LENGTH)
        assert all(b in [0, 1] for b in bits)

    def test_generate_words_correct_length(self):
        gen = DefaultGenerator()
        words = gen.generate_words(LENGTH, 32)
        assert len(words) == LENGTH

    def test_generate_words_in_range(self):
        gen = DefaultGenerator()
        words = gen.generate_words(LENGTH, 32)
        assert all(0 <= w < 2**32 for w in words)

    def test_generate_floats_correct_length(self):
        gen = DefaultGenerator()
        floats = gen.generate_floats(LENGTH, -1.0, 1.0)
        assert len(floats) == LENGTH

    def test_generate_floats_in_range(self):
        gen = DefaultGenerator()
        floats = gen.generate_floats(LENGTH, -1.0, 1.0)
        assert all(-1.0 <= f <= 1.0 for f in floats)

    def test_negative_length_raises(self):
        gen = DefaultGenerator()
        with pytest.raises(ValueError):
            gen.generate_bits(-1)

    def test_zero_length_raises(self):
        gen = DefaultGenerator()
        with pytest.raises(ValueError):
            gen.generate_bits(0)

    def test_seed_reproducibility_bits(self):
        gen1 = DefaultGenerator(seed=42)
        gen2 = DefaultGenerator(seed=42)
        assert gen1.generate_bits(LENGTH) == gen2.generate_bits(LENGTH)

    def test_seed_reproducibility_floats(self):
        gen1 = DefaultGenerator(seed=99)
        gen2 = DefaultGenerator(seed=99)
        assert gen1.generate_floats(LENGTH) == gen2.generate_floats(LENGTH)

    def test_different_seeds_differ(self):
        gen1 = DefaultGenerator(seed=1)
        gen2 = DefaultGenerator(seed=2)
        assert gen1.generate_bits(LENGTH) != gen2.generate_bits(LENGTH)

    def test_reset_restores_sequence(self):
        gen = DefaultGenerator(seed=7)
        bits1 = gen.generate_bits(LENGTH)
        gen.reset()
        bits2 = gen.generate_bits(LENGTH)
        assert bits1 == bits2


class TestLCGGenerator:
    def test_generate_bits_correct_length(self):
        gen = LCGGenerator(seed=12345)
        bits = gen.generate_bits(LENGTH)
        assert len(bits) == LENGTH

    def test_generate_bits_binary(self):
        gen = LCGGenerator(seed=12345)
        bits = gen.generate_bits(LENGTH)
        assert all(b in [0, 1] for b in bits)

    def test_generate_words_correct_length(self):
        gen = LCGGenerator(seed=42)
        words = gen.generate_words(LENGTH, 32)
        assert len(words) == LENGTH

    def test_seed_reproducibility(self):
        gen1 = LCGGenerator(seed=42)
        gen2 = LCGGenerator(seed=42)
        assert gen1.generate_bits(LENGTH) == gen2.generate_bits(LENGTH)

    def test_reset_restores_sequence(self):
        gen = LCGGenerator(seed=99)
        bits1 = gen.generate_bits(LENGTH)
        gen.reset()
        bits2 = gen.generate_bits(LENGTH)
        assert bits1 == bits2

    def test_float_generation(self):
        gen = LCGGenerator(seed=1)
        floats = gen.generate_floats(LENGTH)
        assert len(floats) == LENGTH

    def test_common_park_miller(self):
        gen = CommonLCGGenerators.park_miller(seed=1)
        floats = gen.generate_floats(LENGTH)
        assert len(floats) == LENGTH

    def test_invalid_seed_raises(self):
        modulus = 2**32
        with pytest.raises(ValueError):
            LCGGenerator(modulus=modulus, seed=modulus)

    def test_entropy_in_output(self):
        gen = LCGGenerator(seed=1)
        bits = gen.generate_bits(1000)
        # Should have both 0s and 1s
        assert 0 in bits and 1 in bits


class TestLFSRGenerator:
    def test_fibonacci_generate_bits(self):
        gen = FibonacciLFSRGenerator(width=16, seed=1)
        bits = gen.generate_bits(LENGTH)
        assert len(bits) == LENGTH

    def test_galois_generate_bits(self):
        gen = GaloisLFSRGenerator(width=16, seed=1)
        bits = gen.generate_bits(LENGTH)
        assert len(bits) == LENGTH

    def test_bits_are_binary(self):
        gen = FibonacciLFSRGenerator(width=16, seed=1)
        bits = gen.generate_bits(LENGTH)
        assert all(b in [0, 1] for b in bits)

    def test_seed_reproducibility(self):
        gen1 = FibonacciLFSRGenerator(width=16, seed=5)
        gen2 = FibonacciLFSRGenerator(width=16, seed=5)
        assert gen1.generate_bits(LENGTH) == gen2.generate_bits(LENGTH)

    def test_reset_restores_sequence(self):
        gen = FibonacciLFSRGenerator(width=16, seed=3)
        bits1 = gen.generate_bits(LENGTH)
        gen.reset()
        bits2 = gen.generate_bits(LENGTH)
        assert bits1 == bits2

    def test_entropy_in_output(self):
        gen = FibonacciLFSRGenerator(width=32, seed=1)
        bits = gen.generate_bits(1000)
        assert 0 in bits and 1 in bits

    def test_common_fibonacci_8(self):
        gen = CommonLFSRGenerators.fibonacci_8(seed=1)
        bits = gen.generate_bits(LENGTH)
        assert len(bits) == LENGTH

    def test_invalid_width_raises(self):
        with pytest.raises((ValueError, Exception)):
            LFSRGenerator(width=0)

    def test_zero_seed_raises(self):
        with pytest.raises(ValueError):
            FibonacciLFSRGenerator(width=16, seed=0)


class TestDLFSRGenerator:
    def test_generate_bits(self):
        gen = FibonacciDLFSRGenerator(seed=1)
        bits = gen.generate_bits(LENGTH)
        assert len(bits) == LENGTH

    def test_bits_are_binary(self):
        gen = FibonacciDLFSRGenerator(seed=1)
        bits = gen.generate_bits(LENGTH)
        assert all(b in [0, 1] for b in bits)

    def test_seed_reproducibility(self):
        gen1 = FibonacciDLFSRGenerator(seed=42)
        gen2 = FibonacciDLFSRGenerator(seed=42)
        assert gen1.generate_bits(LENGTH) == gen2.generate_bits(LENGTH)

    def test_reset_restores_sequence(self):
        gen = FibonacciDLFSRGenerator(seed=7)
        bits1 = gen.generate_bits(LENGTH)
        gen.reset()
        bits2 = gen.generate_bits(LENGTH)
        assert bits1 == bits2


class TestLCAGenerator:
    def test_generate_bits(self):
        gen = ElementaryLCAGenerator(seed=1)
        bits = gen.generate_bits(LENGTH)
        assert len(bits) == LENGTH

    def test_bits_are_binary(self):
        gen = ElementaryLCAGenerator(seed=1)
        bits = gen.generate_bits(LENGTH)
        assert all(b in [0, 1] for b in bits)

    def test_seed_reproducibility(self):
        gen1 = ElementaryLCAGenerator(seed=42)
        gen2 = ElementaryLCAGenerator(seed=42)
        assert gen1.generate_bits(LENGTH) == gen2.generate_bits(LENGTH)

    def test_entropy_in_output(self):
        gen = ElementaryLCAGenerator(seed=1)
        bits = gen.generate_bits(1000)
        assert 0 in bits and 1 in bits


class TestPCGGenerator:
    def test_generate_words(self):
        gen = PCGGenerator(seed=42)
        words = gen.generate_words(LENGTH, 32)
        assert len(words) == LENGTH

    def test_generate_bits(self):
        gen = PCGGenerator(seed=42)
        bits = gen.generate_bits(LENGTH)
        assert len(bits) == LENGTH
        assert all(b in [0, 1] for b in bits)

    def test_seed_reproducibility(self):
        gen1 = PCGGenerator(seed=7)
        gen2 = PCGGenerator(seed=7)
        assert gen1.generate_words(LENGTH, 32) == gen2.generate_words(LENGTH, 32)

    def test_reset_restores_sequence(self):
        gen = PCGGenerator(seed=3)
        bits1 = gen.generate_bits(LENGTH)
        gen.reset()
        bits2 = gen.generate_bits(LENGTH)
        assert bits1 == bits2

    def test_entropy_in_output(self):
        gen = PCGGenerator(seed=1)
        words = gen.generate_words(LENGTH, 32)
        assert len(set(words)) > LENGTH // 2


class TestXorshiftGenerators:
    def test_xorshift32_words(self):
        gen = Xorshift32Generator(seed=42)
        words = gen.generate_words(LENGTH, 32)
        assert len(words) == LENGTH
        assert all(0 <= w < 2**32 for w in words)

    def test_xorshift64_words(self):
        gen = Xorshift64Generator(seed=42)
        words = gen.generate_words(LENGTH, 64)
        assert len(words) == LENGTH
        assert all(0 <= w < 2**64 for w in words)

    def test_xorshift32_seed_reproducibility(self):
        gen1 = Xorshift32Generator(seed=99)
        gen2 = Xorshift32Generator(seed=99)
        assert gen1.generate_words(LENGTH, 32) == gen2.generate_words(LENGTH, 32)

    def test_xorshift32_entropy(self):
        gen = Xorshift32Generator(seed=1)
        words = gen.generate_words(LENGTH, 32)
        assert len(set(words)) > LENGTH // 2

    def test_xorshift32_zero_seed_raises(self):
        with pytest.raises(ValueError):
            Xorshift32Generator(seed=0)

    def test_xorshift32_reset(self):
        gen = Xorshift32Generator(seed=5)
        w1 = gen.generate_words(LENGTH, 32)
        gen.reset()
        w2 = gen.generate_words(LENGTH, 32)
        assert w1 == w2

    def test_common_fast_32bit(self):
        gen = CommonXorshiftGenerators.fast_32bit(seed=1)
        words = gen.generate_words(LENGTH, 32)
        assert len(words) == LENGTH


class TestShiftedCounterGenerator:
    """ShiftedCounterGenerator is abstract; tests use the concrete FeistelCounterGenerator."""

    def test_generate_words(self):
        gen = FeistelCounterGenerator(seed=1)
        words = gen.generate_words(LENGTH, 32)
        assert len(words) == LENGTH

    def test_generate_words_in_range(self):
        gen = FeistelCounterGenerator(seed=1)
        words = gen.generate_words(LENGTH, 32)
        assert all(0 <= w < 2**32 for w in words)

    def test_seed_reproducibility(self):
        gen1 = FeistelCounterGenerator(seed=42)
        gen2 = FeistelCounterGenerator(seed=42)
        assert gen1.generate_words(LENGTH, 32) == gen2.generate_words(LENGTH, 32)

    def test_reset_restores_sequence(self):
        gen = FeistelCounterGenerator(seed=7)
        w1 = gen.generate_words(LENGTH, 32)
        gen.reset()
        w2 = gen.generate_words(LENGTH, 32)
        assert w1 == w2

    def test_common_factory(self):
        gen = CommonCounterGenerators.feistel_32bit(seed=1)
        words = gen.generate_words(LENGTH, 32)
        assert len(words) == LENGTH


class TestAdditionalXorshiftVariants:
    def test_xorshift128_words(self):
        from pyhdc.generation.xorshift import Xorshift128Generator
        gen = Xorshift128Generator(seed=42)
        words = gen.generate_words(LENGTH, 32)
        assert len(words) == LENGTH

    def test_xoroshiro_words(self):
        from pyhdc.generation.xorshift import Xoroshiro128PlusGenerator
        gen = Xoroshiro128PlusGenerator(seed=42)
        words = gen.generate_words(LENGTH, 64)
        assert len(words) == LENGTH

    def test_splitmix64_words(self):
        from pyhdc.generation.xorshift import SplitMix64Generator
        gen = SplitMix64Generator(seed=42)
        words = gen.generate_words(LENGTH, 64)
        assert len(words) == LENGTH

    def test_xorshift_balanced_common(self):
        gen = CommonXorshiftGenerators.balanced(seed=1)
        words = gen.generate_words(LENGTH, 32)
        assert len(words) == LENGTH

    def test_xorshift_high_quality_common(self):
        gen = CommonXorshiftGenerators.high_quality(seed=1)
        words = gen.generate_words(LENGTH, 64)
        assert len(words) == LENGTH


class TestAdditionalLCGVariants:
    def test_multiplicative_lcg(self):
        gen = MultiplicativeLCGGenerator(seed=1)
        bits = gen.generate_bits(LENGTH)
        assert len(bits) == LENGTH

    def test_lcg_generate_floats(self):
        gen = LCGGenerator(seed=42)
        floats = gen.generate_floats(LENGTH, 0.0, 1.0)
        assert len(floats) == LENGTH
        assert all(0.0 <= f <= 1.0 for f in floats)


class TestAdditionalPCGVariants:
    def test_multiplicative_pcg(self):
        from pyhdc.generation.pcg import MultiplicativePCGGenerator
        gen = MultiplicativePCGGenerator(seed=42)
        words = gen.generate_words(LENGTH, 32)
        assert len(words) == LENGTH

    def test_common_pcg_pcg32(self):
        gen = CommonPCGGenerators.pcg32(seed=1)
        words = gen.generate_words(LENGTH, 32)
        assert len(words) == LENGTH


class TestCustomGeneratorWithEncoding:
    def test_lcg_with_map_c(self):
        import pyhdc
        gen = LCGGenerator(seed=42)
        enc = pyhdc.MAP_C(dimension=512, generator=gen)
        hv = enc.generate()
        from pyhdc.hypervector import Hypervector
        assert isinstance(hv, Hypervector)
        assert hv.shape == (512,)

    def test_seed_reproducibility_via_encoding(self):
        import pyhdc
        gen1 = LCGGenerator(seed=7)
        gen2 = LCGGenerator(seed=7)
        enc1 = pyhdc.MAP_C(dimension=256, generator=gen1)
        enc2 = pyhdc.MAP_C(dimension=256, generator=gen2)
        np.testing.assert_array_equal(enc1.generate().data, enc2.generate().data)

    def test_lfsr_with_bsc(self):
        import pyhdc
        gen = FibonacciLFSRGenerator(width=16, seed=1)
        enc = pyhdc.BSC(dimension=256, generator=gen)
        hv = enc.generate()
        from pyhdc.hypervector import Hypervector
        assert isinstance(hv, Hypervector)

    def test_xorshift_with_hrr(self):
        import pyhdc
        gen = Xorshift32Generator(seed=42)
        enc = pyhdc.HRR(dimension=256, generator=gen)
        hv = enc.generate()
        from pyhdc.hypervector import Hypervector
        assert isinstance(hv, Hypervector)