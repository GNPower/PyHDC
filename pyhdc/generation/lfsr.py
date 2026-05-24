#!/usr/bin/env python
"""
Linear Feedback Shift Register for HDC

HDC-compatible wrapper for LFSR random number generation.
"""

import random
from typing import Any, Dict, List, Optional

from pyhdc.generation.base import HDCGenerator


class LFSRGenerator(HDCGenerator):
    """
    Linear Feedback Shift Register for hypervector generation.

    Base class for LFSR-based generators with bit-level operations.
    """

    def __init__(
        self,
        width: int = 32,
        taps: Optional[List[int]] = None,
        seed: Optional[int] = None,
        implementation: str = "fibonacci",
    ) -> None:
        """
        Initialize LFSR generator.

        Args:
            width: The width of the LFSR register in bits
            taps: List of tap positions for feedback
            seed: Optional seed for reproducibility
            implementation: Type of LFSR ('fibonacci' or 'galois')
        """
        self._width = width
        self._taps = taps if taps is not None else [width - 1, width - 2]
        self._implementation = implementation
        self._validate_parameters()
        super().__init__(seed)

    def _validate_parameters(self) -> None:
        """Validate LFSR parameters."""
        if not isinstance(self._width, int) or self._width <= 0:
            raise ValueError("Width must be a positive integer")

        if not isinstance(self._taps, list) or not all(
            isinstance(t, int) for t in self._taps
        ):
            raise ValueError("Taps must be a list of integers")

        if not all(0 <= t < self._width for t in self._taps):
            raise ValueError(f"Taps must be between 0 and {self._width - 1}")

        if len(self._taps) == 0:
            raise ValueError("At least one tap position must be specified")

    def _configure_internal(self) -> None:
        """Configure the LFSR state."""
        self._max_value = (1 << self._width) - 1

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

    def _next_bit_fibonacci(self) -> int:
        """Generate next bit using Fibonacci configuration."""
        feedback_bit = 0
        for tap in self._taps_sorted:
            feedback_bit ^= (self._state >> tap) & 1

        output_bit = self._state & 1
        self._state = (self._state >> 1) | (feedback_bit << (self._width - 1))

        return output_bit

    def _next_bit_galois(self) -> int:
        """Generate next bit using Galois configuration."""
        output_bit = self._state & 1
        self._state >>= 1

        if output_bit:
            for tap in self._taps_sorted:
                self._state ^= 1 << tap

        return output_bit

    def _next_bit(self) -> int:
        """Generate next bit."""
        if self._implementation == "fibonacci":
            return self._next_bit_fibonacci()
        elif self._implementation == "galois":
            return self._next_bit_galois()
        else:
            raise ValueError(f"Unknown implementation: {self._implementation}")

    def _next_word(self, word_size: int = 32) -> int:
        """Generate next word by collecting bits."""
        word = 0
        for i in range(word_size):
            word |= self._next_bit() << i
        return word

    def set_parameters(
        self,
        width: Optional[int] = None,
        taps: Optional[List[int]] = None,
        seed: Optional[int] = None,
        implementation: Optional[str] = None,
    ) -> None:
        """
        Set LFSR parameters.

        Args:
            width: The width of the LFSR register in bits
            taps: List of tap positions
            seed: The seed value
            implementation: Type of LFSR
        """
        if width is not None:
            self._width = width

        if taps is not None:
            self._taps = taps

        if implementation is not None:
            if implementation not in ["fibonacci", "galois"]:
                raise ValueError("Implementation must be 'fibonacci' or 'galois'")
            self._implementation = implementation

        if seed is not None:
            self._seed = seed

        self._validate_parameters()
        self._configure_internal()

    def set_width(self, width: int) -> None:
        """Set the width parameter."""
        self.set_parameters(width=width)

    def set_taps(self, taps: List[int]) -> None:
        """Set the tap positions."""
        self.set_parameters(taps=taps)

    def set_implementation(self, implementation: str) -> None:
        """Set the LFSR implementation type."""
        self.set_parameters(implementation=implementation)

    def get_parameters(self) -> Dict[str, Any]:
        """Get current LFSR parameters."""
        return {
            "width": self._width,
            "taps": self._taps.copy(),
            "seed": self._seed,
            "implementation": self._implementation,
        }

    def get_width(self) -> int:
        """Get the width parameter."""
        return self._width

    def get_taps(self) -> List[int]:
        """Get the tap positions."""
        return self._taps.copy()

    def get_implementation(self) -> str:
        """Get the implementation type."""
        return self._implementation

    def reset(self) -> None:
        """Reset to initial state."""
        self._state = self._initial_state

    def get_state(self) -> int:
        """Get current state."""
        return self._state


class FibonacciLFSRGenerator(LFSRGenerator):
    """Fibonacci-style LFSR generator."""

    def __init__(
        self,
        width: int = 32,
        taps: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Fibonacci LFSR."""
        super().__init__(width, taps, seed, "fibonacci")


class GaloisLFSRGenerator(LFSRGenerator):
    """Galois-style LFSR generator."""

    def __init__(
        self,
        width: int = 32,
        taps: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Galois LFSR."""
        super().__init__(width, taps, seed, "galois")


# Predefined LFSR configurations
class CommonLFSRGenerators:
    """Factory for common LFSR configurations."""

    @staticmethod
    def fibonacci_8(seed: Optional[int] = None) -> FibonacciLFSRGenerator:
        """Standard 8-bit Fibonacci LFSR."""
        return FibonacciLFSRGenerator(width=8, taps=[7, 5, 4, 3], seed=seed)

    @staticmethod
    def fibonacci_16(seed: Optional[int] = None) -> FibonacciLFSRGenerator:
        """Standard 16-bit Fibonacci LFSR."""
        return FibonacciLFSRGenerator(width=16, taps=[15, 14, 12, 3], seed=seed)

    @staticmethod
    def fibonacci_32(seed: Optional[int] = None) -> FibonacciLFSRGenerator:
        """Standard 32-bit Fibonacci LFSR."""
        return FibonacciLFSRGenerator(width=32, taps=[31, 21, 1, 0], seed=seed)

    @staticmethod
    def galois_16(seed: Optional[int] = None) -> GaloisLFSRGenerator:
        """Standard 16-bit Galois LFSR."""
        return GaloisLFSRGenerator(width=16, taps=[15, 14, 12, 3], seed=seed)

    @staticmethod
    def galois_32(seed: Optional[int] = None) -> GaloisLFSRGenerator:
        """Standard 32-bit Galois LFSR."""
        return GaloisLFSRGenerator(width=32, taps=[31, 21, 1, 0], seed=seed)

    @staticmethod
    def maximal_64(seed: Optional[int] = None) -> FibonacciLFSRGenerator:
        """64-bit LFSR with maximal period taps."""
        return FibonacciLFSRGenerator(width=64, taps=[63, 62, 60, 59], seed=seed)


# Legacy alias for backward compatibility
LFSR = FibonacciLFSRGenerator
