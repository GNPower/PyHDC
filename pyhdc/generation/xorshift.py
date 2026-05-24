#!/usr/bin/env python
"""
Xorshift Family Generators for HDC

HDC-compatible wrapper for Xorshift random number generation.
"""

import random
from typing import Any, Dict, List, Optional, Union

from pyhdc.generation.base import HDCGenerator


class XorshiftGenerator(HDCGenerator):
    """
    Base class for Xorshift family generators.
    
    Xorshift generators use XOR and shift operations for fast pseudorandom
    number generation with good statistical properties.
    """
    
    def __init__(
        self,
        state_size: int = 1,
        word_bits: int = 64,
        seed: Optional[Union[int, List[int]]] = None
    ) -> None:
        """
        Initialize Xorshift generator.
        
        Args:
            state_size: Number of words in state
            word_bits: Bits per word (32 or 64)
            seed: Optional seed (int or list of ints)
        """
        self._state_size = state_size
        self._word_bits = word_bits
        self._word_mask = (1 << word_bits) - 1
        super().__init__(seed)
    
    def _configure_internal(self) -> None:
        """Configure the Xorshift state."""
        if self._seed is None:
            self._state = [random.randint(1, self._word_mask) for _ in range(self._state_size)]
        elif isinstance(self._seed, int):
            if self._seed == 0:
                raise ValueError("Seed cannot be zero")
            # Split integer seed into multiple words
            self._state = []
            for i in range(self._state_size):
                word = (self._seed >> (i * self._word_bits)) & self._word_mask
                if word == 0:
                    word = (self._seed % (self._word_mask - 1)) + 1
                self._state.append(word)
        elif isinstance(self._seed, list):
            if len(self._seed) != self._state_size:
                raise ValueError(f"Seed list must have {self._state_size} elements")
            if all(s == 0 for s in self._seed):
                raise ValueError("Seed cannot be all zeros")
            self._state = [s & self._word_mask for s in self._seed]
        else:
            raise ValueError("Seed must be int, list of ints, or None")
        
        self._initial_state = self._state.copy()
    
    def _next_value(self) -> int:
        """Generate next value (implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _next_value")
    
    def _next_bit(self) -> int:
        """Generate next bit."""
        return self._next_value() & 1
    
    def _next_word(self, word_size: int = 32) -> int:
        """Generate next word."""
        if word_size <= self._word_bits:
            return self._next_value() & ((1 << word_size) - 1)
        else:
            # Combine multiple outputs
            result = 0
            bits_collected = 0
            while bits_collected < word_size:
                value = self._next_value()
                result |= (value << bits_collected)
                bits_collected += self._word_bits
            return result & ((1 << word_size) - 1)
    
    def set_parameters(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        **kwargs
    ) -> None:
        """Set Xorshift parameters."""
        if seed is not None:
            self._seed = seed
        self._configure_internal()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {
            "state_size": self._state_size,
            "word_bits": self._word_bits,
            "seed": self._seed
        }
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._state = self._initial_state.copy()
    
    def get_state(self) -> List[int]:
        """Get current state."""
        return self._state.copy()


class Xorshift32Generator(XorshiftGenerator):
    """32-bit Xorshift generator with (13,17,5) parameters."""
    
    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize Xorshift32."""
        super().__init__(state_size=1, word_bits=32, seed=seed)
    
    def _next_value(self) -> int:
        """Generate next 32-bit value."""
        x = self._state[0]
        x ^= (x << 13) & self._word_mask
        x ^= (x >> 17) & self._word_mask
        x ^= (x << 5) & self._word_mask
        self._state[0] = x
        return x


class Xorshift64Generator(XorshiftGenerator):
    """64-bit Xorshift generator with (13,7,17) parameters."""
    
    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize Xorshift64."""
        super().__init__(state_size=1, word_bits=64, seed=seed)
    
    def _next_value(self) -> int:
        """Generate next 64-bit value."""
        x = self._state[0]
        x ^= (x << 13) & self._word_mask
        x ^= (x >> 7) & self._word_mask
        x ^= (x << 17) & self._word_mask
        self._state[0] = x
        return x


class Xorshift128Generator(XorshiftGenerator):
    """128-bit Xorshift generator (4 x 32-bit words)."""
    
    def __init__(self, seed: Optional[Union[int, List[int]]] = None) -> None:
        """Initialize Xorshift128."""
        super().__init__(state_size=4, word_bits=32, seed=seed)
    
    def _next_value(self) -> int:
        """Generate next 32-bit value."""
        t = self._state[3]
        s = self._state[0]
        
        self._state[3] = self._state[2]
        self._state[2] = self._state[1]
        self._state[1] = s
        
        t ^= (t << 11) & self._word_mask
        t ^= (t >> 8) & self._word_mask
        t ^= s ^ ((s >> 19) & self._word_mask)
        
        self._state[0] = t
        return t


class XorshiftPlusGenerator(XorshiftGenerator):
    """Xorshift+ generator with addition for improved low-bit randomness."""
    
    def __init__(self, seed: Optional[Union[int, List[int]]] = None) -> None:
        """Initialize Xorshift+."""
        super().__init__(state_size=2, word_bits=64, seed=seed)
    
    def _next_value(self) -> int:
        """Generate next value using xorshift+ algorithm."""
        x = self._state[0]
        y = self._state[1]
        
        self._state[0] = y
        x ^= (x << 23) & self._word_mask
        self._state[1] = x ^ y ^ ((x >> 17) & self._word_mask) ^ ((y >> 26) & self._word_mask)
        
        return (self._state[1] + y) & self._word_mask


class XorshiftStarGenerator(XorshiftGenerator):
    """Xorshift* generator with multiplication for better statistics."""
    
    def __init__(self, seed: Optional[Union[int, List[int]]] = None) -> None:
        """Initialize Xorshift*."""
        super().__init__(state_size=2, word_bits=64, seed=seed)
        self._multiplier = 0x2545F4914F6CDD1D
    
    def _next_value(self) -> int:
        """Generate next value using xorshift* algorithm."""
        x = self._state[0]
        y = self._state[1]
        
        self._state[0] = y
        x ^= (x << 23) & self._word_mask
        self._state[1] = x ^ y ^ ((x >> 17) & self._word_mask) ^ ((y >> 26) & self._word_mask)
        
        return (self._state[1] * self._multiplier) & self._word_mask


class Xoshiro256StarStarGenerator(XorshiftGenerator):
    """Xoshiro256** - modern xorshift with excellent properties."""
    
    def __init__(self, seed: Optional[Union[int, List[int]]] = None) -> None:
        """Initialize Xoshiro256**."""
        super().__init__(state_size=4, word_bits=64, seed=seed)
    
    def _rotl64(self, x: int, k: int) -> int:
        """Rotate 64-bit integer left."""
        k &= 63
        return ((x << k) | (x >> (64 - k))) & self._word_mask
    
    def _next_value(self) -> int:
        """Generate next value using xoshiro256** algorithm."""
        result = self._rotl64(self._state[1] * 5, 7) * 9
        result &= self._word_mask
        
        t = (self._state[1] << 17) & self._word_mask
        
        self._state[2] ^= self._state[0]
        self._state[3] ^= self._state[1]
        self._state[1] ^= self._state[2]
        self._state[0] ^= self._state[3]
        
        self._state[2] ^= t
        self._state[3] = self._rotl64(self._state[3], 45)
        
        return result
    
    def jump(self) -> None:
        """Jump ahead by 2^128 steps."""
        JUMP = [0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 
                0xa9582618e03fc9aa, 0x39abdc4529b1661c]
        
        s0 = s1 = s2 = s3 = 0
        for jump_val in JUMP:
            for b in range(64):
                if jump_val & (1 << b):
                    s0 ^= self._state[0]
                    s1 ^= self._state[1]
                    s2 ^= self._state[2]
                    s3 ^= self._state[3]
                self._next_value()
        
        self._state[0] = s0
        self._state[1] = s1
        self._state[2] = s2
        self._state[3] = s3


class Xoroshiro128PlusGenerator(XorshiftGenerator):
    """Xoroshiro128+ - fast variant with 128-bit state."""
    
    def __init__(self, seed: Optional[Union[int, List[int]]] = None) -> None:
        """Initialize Xoroshiro128+."""
        super().__init__(state_size=2, word_bits=64, seed=seed)
    
    def _rotl64(self, x: int, k: int) -> int:
        """Rotate 64-bit integer left."""
        k &= 63
        return ((x << k) | (x >> (64 - k))) & self._word_mask
    
    def _next_value(self) -> int:
        """Generate next value using xoroshiro128+ algorithm."""
        s0 = self._state[0]
        s1 = self._state[1]
        result = (s0 + s1) & self._word_mask
        
        s1 ^= s0
        self._state[0] = self._rotl64(s0, 24) ^ s1 ^ ((s1 << 16) & self._word_mask)
        self._state[1] = self._rotl64(s1, 37)
        
        return result
    
    def jump(self) -> None:
        """Jump ahead by 2^64 steps."""
        JUMP = [0xdf900294d8f554a5, 0x170865df4b3201fc]
        
        s0 = s1 = 0
        for jump_val in JUMP:
            for b in range(64):
                if jump_val & (1 << b):
                    s0 ^= self._state[0]
                    s1 ^= self._state[1]
                self._next_value()
        
        self._state[0] = s0
        self._state[1] = s1


class Xoroshiro128StarStarGenerator(XorshiftGenerator):
    """Xoroshiro128** - improved statistics over Plus variant."""
    
    def __init__(self, seed: Optional[Union[int, List[int]]] = None) -> None:
        """Initialize Xoroshiro128**."""
        super().__init__(state_size=2, word_bits=64, seed=seed)
    
    def _rotl64(self, x: int, k: int) -> int:
        """Rotate 64-bit integer left."""
        k &= 63
        return ((x << k) | (x >> (64 - k))) & self._word_mask
    
    def _next_value(self) -> int:
        """Generate next value using xoroshiro128** algorithm."""
        s0 = self._state[0]
        s1 = self._state[1]
        result = self._rotl64(s0 * 5, 7) * 9
        result &= self._word_mask
        
        s1 ^= s0
        self._state[0] = self._rotl64(s0, 24) ^ s1 ^ ((s1 << 16) & self._word_mask)
        self._state[1] = self._rotl64(s1, 37)
        
        return result


class SplitMix64Generator(XorshiftGenerator):
    """SplitMix64 - good for seeding other generators."""
    
    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize SplitMix64."""
        super().__init__(state_size=1, word_bits=64, seed=seed)
        self._gamma = 0x9e3779b97f4a7c15
    
    def _next_value(self) -> int:
        """Generate next value using SplitMix64 algorithm."""
        self._state[0] = (self._state[0] + self._gamma) & self._word_mask
        z = self._state[0]
        
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
        z &= self._word_mask
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb
        z &= self._word_mask
        z = z ^ (z >> 31)
        
        return z


# Utility for seeding complex generators
def splitmix64_seed(initial_seed: int, count: int) -> List[int]:
    """
    Use SplitMix64 to generate multiple seed values.
    
    Args:
        initial_seed: Initial seed value
        count: Number of seed values to generate
        
    Returns:
        List of seed values
    """
    splitmix = SplitMix64Generator(initial_seed)
    return splitmix.generate_words(count, 64)


# Predefined Xorshift configurations
class CommonXorshiftGenerators:
    """Factory for common Xorshift configurations."""
    
    @staticmethod
    def fast_32bit(seed: Optional[int] = None) -> Xorshift32Generator:
        """Fast 32-bit generator for simple applications."""
        return Xorshift32Generator(seed)
    
    @staticmethod
    def standard_64bit(seed: Optional[int] = None) -> Xorshift64Generator:
        """Standard 64-bit generator."""
        return Xorshift64Generator(seed)
    
    @staticmethod
    def high_quality(seed: Optional[int] = None) -> Xoshiro256StarStarGenerator:
        """High-quality generator recommended for general use."""
        if seed is None:
            return Xoshiro256StarStarGenerator()
        else:
            seeds = splitmix64_seed(seed, 4)
            return Xoshiro256StarStarGenerator(seeds)
    
    @staticmethod
    def fastest(seed: Optional[int] = None) -> Xoroshiro128PlusGenerator:
        """Fastest generator in the family."""
        if seed is None:
            return Xoroshiro128PlusGenerator()
        else:
            seeds = splitmix64_seed(seed, 2)
            return Xoroshiro128PlusGenerator(seeds)
    
    @staticmethod
    def balanced(seed: Optional[int] = None) -> Xoroshiro128StarStarGenerator:
        """Balanced speed/quality generator."""
        if seed is None:
            return Xoroshiro128StarStarGenerator()
        else:
            seeds = splitmix64_seed(seed, 2)
            return Xoroshiro128StarStarGenerator(seeds)
    
    @staticmethod
    def for_seeding(seed: Optional[int] = None) -> SplitMix64Generator:
        """Generator optimized for seeding other generators."""
        return SplitMix64Generator(seed)


# Legacy aliases for backward compatibility
Xorshift = Xorshift64Generator
XorshiftStar64 = XorshiftStarGenerator
XorshiftPlus64 = XorshiftPlusGenerator
