#!/usr/bin/env python
"""
Digit-Serial Linear Feedback Shift Register for HDC

HDC-compatible wrapper for DLFSR random number generation.
"""

import random
from typing import Any, Dict, List, Optional

from pyhdc.generation.base import HDCGenerator


class DLFSRGenerator(HDCGenerator):
    """
    Digit-Serial Linear Feedback Shift Register for hypervector generation.

    Base class for DLFSR-based generators with word-level operations.
    """

    def __init__(
        self,
        width: int = 32,
        word_size: int = 32,
        taps: Optional[List[int]] = None,
        seed: Optional[int] = None,
        implementation: str = "fibonacci",
    ) -> None:
        """
        Initialize DLFSR generator.

        Args:
            width: The width of the DLFSR register in words
            word_size: The size of each word in bits
            taps: List of tap positions for feedback
            seed: Optional seed for reproducibility
            implementation: Type of DLFSR ('fibonacci', 'galois', 'matrix')
        """
        self._width = width
        self._word_size = word_size
        self._taps = taps if taps is not None else [width - 1, width - 2]
        self._implementation = implementation
        self._validate_parameters()
        super().__init__(seed)

    def _validate_parameters(self) -> None:
        """Validate DLFSR parameters."""
        if not isinstance(self._width, int) or self._width <= 0:
            raise ValueError("Width must be a positive integer")

        if not isinstance(self._word_size, int) or self._word_size <= 0:
            raise ValueError("Word size must be a positive integer")

        if not isinstance(self._taps, list) or not all(
            isinstance(t, int) for t in self._taps
        ):
            raise ValueError("Taps must be a list of integers")

        if not all(0 <= t < self._width for t in self._taps):
            raise ValueError(f"Taps must be between 0 and {self._width - 1}")

        if len(self._taps) == 0:
            raise ValueError("At least one tap position must be specified")

    def _configure_internal(self) -> None:
        """Configure the DLFSR state."""
        self._word_mask = (1 << self._word_size) - 1
        self._total_bits = self._width * self._word_size
        self._max_value = (1 << self._total_bits) - 1

        if self._seed is None:
            self._state = random.randint(1, self._max_value)
        else:
            if (
                not isinstance(self._seed, int)
                or self._seed <= 0
                or self._seed > self._max_value
            ):
                raise ValueError(f"Seed must be between 1 and {self._max_value}")
            self._state = self._seed

        self._initial_state = self._state
        self._taps_sorted = sorted(list(set(self._taps)), reverse=True)

        # Buffer for bit generation
        self._bit_buffer = 0
        self._bits_available = 0

    def _get_word(self, position: int) -> int:
        """Extract a word from the register at given position."""
        return (self._state >> (position * self._word_size)) & self._word_mask

    def _set_word(self, position: int, value: int) -> None:
        """Set a word in the register at given position."""
        clear_mask = ~(self._word_mask << (position * self._word_size))
        self._state &= clear_mask
        masked_value = value & self._word_mask
        self._state |= masked_value << (position * self._word_size)

    def _next_word_fibonacci(self) -> int:
        """Generate next word using Fibonacci configuration."""
        feedback_word = 0
        for tap in self._taps_sorted:
            feedback_word ^= self._get_word(tap)

        output_word = self._get_word(0)
        self._state >>= self._word_size
        self._set_word(self._width - 1, feedback_word)

        return output_word

    def _next_word_galois(self) -> int:
        """Generate next word using Galois configuration."""
        output_word = self._get_word(0)
        self._state >>= self._word_size

        if output_word:
            for tap in self._taps_sorted:
                if tap > 0:
                    current_word = self._get_word(tap - 1)
                    self._set_word(tap - 1, current_word ^ output_word)

        return output_word

    def _next_word(self, word_size: int = 32) -> int:
        """Generate next word."""
        if self._implementation == "fibonacci":
            word = self._next_word_fibonacci()
        elif self._implementation == "galois":
            word = self._next_word_galois()
        else:
            raise ValueError(f"Unknown implementation: {self._implementation}")

        # If requested word size differs from DLFSR word size
        if word_size != self._word_size:
            if word_size < self._word_size:
                return word & ((1 << word_size) - 1)
            else:
                # Combine multiple words
                result = word
                remaining = word_size - self._word_size
                while remaining > 0:
                    next_word = self._next_word(self._word_size)
                    result = (result << self._word_size) | next_word
                    remaining -= self._word_size
                return result & ((1 << word_size) - 1)

        return word

    def _next_bit(self) -> int:
        """Generate next bit."""
        if self._bits_available == 0:
            self._bit_buffer = self._next_word(self._word_size)
            self._bits_available = self._word_size

        bit = self._bit_buffer & 1
        self._bit_buffer >>= 1
        self._bits_available -= 1

        return bit

    def set_parameters(
        self,
        width: Optional[int] = None,
        word_size: Optional[int] = None,
        taps: Optional[List[int]] = None,
        seed: Optional[int] = None,
        implementation: Optional[str] = None,
    ) -> None:
        """
        Set DLFSR parameters.

        Args:
            width: The width of the DLFSR register in words
            word_size: The size of each word in bits
            taps: List of tap positions
            seed: The seed value
            implementation: Type of DLFSR
        """
        if width is not None:
            self._width = width

        if word_size is not None:
            self._word_size = word_size

        if taps is not None:
            self._taps = taps

        if implementation is not None:
            if implementation not in ["fibonacci", "galois", "matrix"]:
                raise ValueError(
                    "Implementation must be 'fibonacci', 'galois', or 'matrix'"
                )
            self._implementation = implementation

        if seed is not None:
            self._seed = seed

        self._validate_parameters()
        self._configure_internal()

    def set_width(self, width: int) -> None:
        """Set the width parameter."""
        self.set_parameters(width=width)

    def set_word_size(self, word_size: int) -> None:
        """Set the word size parameter."""
        self.set_parameters(word_size=word_size)

    def set_taps(self, taps: List[int]) -> None:
        """Set the tap positions."""
        self.set_parameters(taps=taps)

    def set_implementation(self, implementation: str) -> None:
        """Set the DLFSR implementation type."""
        self.set_parameters(implementation=implementation)

    def get_parameters(self) -> Dict[str, Any]:
        """Get current DLFSR parameters."""
        return {
            "width": self._width,
            "word_size": self._word_size,
            "taps": self._taps.copy(),
            "seed": self._seed,
            "implementation": self._implementation,
        }

    def get_width(self) -> int:
        """Get the width parameter."""
        return self._width

    def get_word_size(self) -> int:
        """Get the word size parameter."""
        return self._word_size

    def get_taps(self) -> List[int]:
        """Get the tap positions."""
        return self._taps.copy()

    def get_implementation(self) -> str:
        """Get the implementation type."""
        return self._implementation

    def reset(self) -> None:
        """Reset to initial state."""
        self._state = self._initial_state
        self._bit_buffer = 0
        self._bits_available = 0

    def get_state(self) -> int:
        """Get current state."""
        return self._state


class FibonacciDLFSRGenerator(DLFSRGenerator):
    """Fibonacci-style DLFSR generator."""

    def __init__(
        self,
        width: int = 32,
        word_size: int = 32,
        taps: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Fibonacci DLFSR."""
        super().__init__(width, word_size, taps, seed, "fibonacci")


class GaloisDLFSRGenerator(DLFSRGenerator):
    """Galois-style DLFSR generator."""

    def __init__(
        self,
        width: int = 32,
        word_size: int = 32,
        taps: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Galois DLFSR."""
        super().__init__(width, word_size, taps, seed, "galois")


class MatrixDLFSRGenerator(DLFSRGenerator):
    """
    Matrix-based DLFSR generator.

    Uses matrix multiplication over GF(2^word_size) for feedback.
    """

    def __init__(
        self,
        width: int = 32,
        word_size: int = 32,
        taps: Optional[List[int]] = None,
        matrix_coeffs: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Matrix DLFSR."""
        self._matrix_coeffs = matrix_coeffs
        super().__init__(width, word_size, taps, seed, "matrix")

    def _configure_internal(self) -> None:
        """Configure with matrix coefficients."""
        super()._configure_internal()

        if self._matrix_coeffs is None:
            self._matrix_coeffs = [1] * len(self._taps)
        else:
            if len(self._matrix_coeffs) != len(self._taps):
                raise ValueError("Matrix coefficients must match number of taps")

    def _galois_multiply(self, a: int, b: int) -> int:
        """Multiply in GF(2^word_size)."""
        result = 0
        for i in range(self._word_size):
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & (1 << self._word_size):
                # Simple irreducible polynomial: x^word_size + x + 1
                a ^= (1 << self._word_size) | 3
        return result & self._word_mask

    def _next_word(self, word_size: int = 32) -> int:
        """Generate next word using matrix feedback."""
        feedback_word = 0
        for tap, coeff in zip(self._taps_sorted, self._matrix_coeffs):
            word_value = self._get_word(tap)
            feedback_word ^= self._galois_multiply(word_value, coeff)

        output_word = self._get_word(0)
        self._state >>= self._word_size
        self._set_word(self._width - 1, feedback_word)

        if word_size != self._word_size:
            if word_size < self._word_size:
                return output_word & ((1 << word_size) - 1)
            else:
                result = output_word
                remaining = word_size - self._word_size
                while remaining > 0:
                    next_word = self._next_word(self._word_size)
                    result = (result << self._word_size) | next_word
                    remaining -= self._word_size
                return result & ((1 << word_size) - 1)

        return output_word

    def set_matrix_coefficients(self, coeffs: List[int]) -> None:
        """Set matrix coefficients."""
        if len(coeffs) != len(self._taps):
            raise ValueError("Coefficients must match number of taps")
        self._matrix_coeffs = coeffs.copy()

    def get_matrix_coefficients(self) -> List[int]:
        """Get matrix coefficients."""
        return self._matrix_coeffs.copy()

    def set_parameters(
        self,
        width: Optional[int] = None,
        word_size: Optional[int] = None,
        taps: Optional[List[int]] = None,
        matrix_coeffs: Optional[List[int]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Set parameters including matrix coefficients."""
        if matrix_coeffs is not None:
            self._matrix_coeffs = matrix_coeffs
        super().set_parameters(width, word_size, taps, seed, "matrix")

    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters including matrix coefficients."""
        params = super().get_parameters()
        params["matrix_coeffs"] = self._matrix_coeffs.copy()
        return params


# Predefined DLFSR configurations
class CommonDLFSRGenerators:
    """Factory for common DLFSR configurations."""

    @staticmethod
    def fibonacci_32(seed: Optional[int] = None) -> FibonacciDLFSRGenerator:
        """Standard 32-word Fibonacci DLFSR."""
        return FibonacciDLFSRGenerator(
            width=32, word_size=32, taps=[31, 30, 29, 1], seed=seed
        )

    @staticmethod
    def galois_32(seed: Optional[int] = None) -> GaloisDLFSRGenerator:
        """Standard 32-word Galois DLFSR."""
        return GaloisDLFSRGenerator(
            width=32, word_size=32, taps=[31, 30, 29, 1], seed=seed
        )

    @staticmethod
    def fibonacci_64(seed: Optional[int] = None) -> FibonacciDLFSRGenerator:
        """64-word Fibonacci DLFSR for longer sequences."""
        return FibonacciDLFSRGenerator(
            width=64, word_size=32, taps=[63, 62, 60, 59], seed=seed
        )

    @staticmethod
    def compact_8(seed: Optional[int] = None) -> FibonacciDLFSRGenerator:
        """Compact 8-word DLFSR for fast generation."""
        return FibonacciDLFSRGenerator(
            width=8, word_size=32, taps=[7, 5, 4, 3], seed=seed
        )


# Legacy alias for backward compatibility
DLFSR = FibonacciDLFSRGenerator
