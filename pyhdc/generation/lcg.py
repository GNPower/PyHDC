#!/usr/bin/env python
"""
Linear Congruential Generator for HDC

HDC-compatible wrapper for LCG random number generation.
"""

import random
from typing import Any, Dict, List, Optional

from pyhdc.generation.base import HDCGenerator


class LCGGenerator(HDCGenerator):
    """
    Linear Congruential Generator for hypervector generation.
    
    Uses the formula: X(n+1) = (a * X(n) + c) mod m
    where a is the multiplier, c is the increment, and m is the modulus.
    """
    
    def __init__(
        self,
        modulus: int = 2**32,
        multiplier: int = 1664525,
        increment: int = 1013904223,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize LCG generator.
        
        Args:
            modulus: The modulus value (m)
            multiplier: The multiplier value (a)
            increment: The increment value (c)
            seed: Optional seed for reproducibility
        """
        self._modulus = modulus
        self._multiplier = multiplier
        self._increment = increment
        super().__init__(seed)
    
    def _configure_internal(self) -> None:
        """Configure the LCG state."""
        if self._seed is None:
            self._state = random.randint(0, self._modulus - 1)
        else:
            if not isinstance(self._seed, int) or self._seed < 0 or self._seed >= self._modulus:
                raise ValueError(f"Seed must be between 0 and {self._modulus - 1}")
            self._state = self._seed
        self._initial_state = self._state
    
    def _next_value(self) -> int:
        """Generate next LCG value."""
        self._state = (self._multiplier * self._state + self._increment) % self._modulus
        return self._state
    
    def _next_bit(self) -> int:
        """Generate next bit from LCG."""
        value = self._next_value()
        # Use least significant bit
        return value & 1
    
    def _next_word(self, word_size: int = 32) -> int:
        """Generate next word from LCG."""
        if word_size > 32:
            # For larger words, combine multiple LCG outputs
            result = 0
            for i in range((word_size + 31) // 32):
                result |= self._next_value() << (i * 32)
            return result & ((1 << word_size) - 1)
        else:
            # Mask to desired word size
            return self._next_value() & ((1 << word_size) - 1)
    
    def generate_floats(self, length: int, min_val: float = -1.0, 
                       max_val: float = 1.0) -> List[float]:
        """Generate floats efficiently using LCG."""
        result = []
        for _ in range(length):
            value = self._next_value()
            # Normalize to [0, 1]
            normalized = value / self._modulus
            # Scale to [min_val, max_val]
            result.append(min_val + normalized * (max_val - min_val))
        return result
    
    def set_parameters(
        self,
        modulus: Optional[int] = None,
        multiplier: Optional[int] = None,
        increment: Optional[int] = None,
        seed: Optional[int] = None
    ) -> None:
        """
        Set LCG parameters.
        
        Args:
            modulus: The modulus value (m)
            multiplier: The multiplier value (a)
            increment: The increment value (c)
            seed: The seed value
        """
        if modulus is not None:
            if not isinstance(modulus, int) or modulus <= 1:
                raise ValueError("Modulus must be a positive integer greater than 1")
            self._modulus = modulus
        
        if multiplier is not None:
            if not isinstance(multiplier, int) or multiplier < 0 or multiplier >= self._modulus:
                raise ValueError(f"Multiplier must be between 0 and {self._modulus - 1}")
            self._multiplier = multiplier
        
        if increment is not None:
            if not isinstance(increment, int) or increment < 0 or increment >= self._modulus:
                raise ValueError(f"Increment must be between 0 and {self._modulus - 1}")
            self._increment = increment
        
        if seed is not None:
            self._seed = seed
        
        # Reconfigure with new parameters
        self._configure_internal()
    
    def set_modulus(self, modulus: int) -> None:
        """Set the modulus parameter."""
        self.set_parameters(modulus=modulus)
    
    def set_multiplier(self, multiplier: int) -> None:
        """Set the multiplier parameter."""
        self.set_parameters(multiplier=multiplier)
    
    def set_increment(self, increment: int) -> None:
        """Set the increment parameter."""
        self.set_parameters(increment=increment)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current LCG parameters."""
        return {
            "modulus": self._modulus,
            "multiplier": self._multiplier,
            "increment": self._increment,
            "seed": self._seed
        }
    
    def get_modulus(self) -> int:
        """Get the modulus parameter."""
        return self._modulus
    
    def get_multiplier(self) -> int:
        """Get the multiplier parameter."""
        return self._multiplier
    
    def get_increment(self) -> int:
        """Get the increment parameter."""
        return self._increment
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._state = self._initial_state
    
    def get_state(self) -> int:
        """Get current state."""
        return self._state


class MultiplicativeLCGGenerator(LCGGenerator):
    """
    Multiplicative LCG Generator (increment = 0).
    
    Uses the formula: X(n+1) = (a * X(n)) mod m
    """
    
    def __init__(
        self,
        modulus: int = 2**31 - 1,
        multiplier: int = 48271,
        seed: Optional[int] = None
    ) -> None:
        """Initialize multiplicative LCG."""
        super().__init__(modulus, multiplier, 0, seed)
    
    def _configure_internal(self) -> None:
        """Configure with non-zero seed."""
        if self._seed is None:
            self._state = random.randint(1, self._modulus - 1)
        else:
            if not isinstance(self._seed, int) or self._seed <= 0 or self._seed >= self._modulus:
                raise ValueError(f"Seed must be between 1 and {self._modulus - 1}")
            self._state = self._seed
        self._initial_state = self._state
    
    def set_parameters(
        self,
        modulus: Optional[int] = None,
        multiplier: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> None:
        """Set parameters (increment is always 0)."""
        if "increment" in kwargs and kwargs["increment"] != 0:
            raise ValueError("Multiplicative LCG always has increment = 0")
        super().set_parameters(modulus, multiplier, 0, seed)


# Predefined LCG configurations
class CommonLCGGenerators:
    """Factory for common LCG configurations."""
    
    @staticmethod
    def numerical_recipes(seed: Optional[int] = None) -> LCGGenerator:
        """Numerical Recipes LCG."""
        return LCGGenerator(
            modulus=2**32,
            multiplier=1664525,
            increment=1013904223,
            seed=seed
        )
    
    @staticmethod
    def park_miller(seed: Optional[int] = None) -> MultiplicativeLCGGenerator:
        """Park-Miller minimal standard LCG."""
        return MultiplicativeLCGGenerator(
            modulus=2**31 - 1,
            multiplier=16807,
            seed=seed
        )
    
    @staticmethod
    def minstd(seed: Optional[int] = None) -> MultiplicativeLCGGenerator:
        """MINSTD (improved Park-Miller) LCG."""
        return MultiplicativeLCGGenerator(
            modulus=2**31 - 1,
            multiplier=48271,
            seed=seed
        )
    
    @staticmethod
    def borland(seed: Optional[int] = None) -> LCGGenerator:
        """Borland C/C++ LCG."""
        return LCGGenerator(
            modulus=2**32,
            multiplier=22695477,
            increment=1,
            seed=seed
        )
    
    @staticmethod
    def msvc(seed: Optional[int] = None) -> LCGGenerator:
        """Microsoft Visual C++ LCG."""
        return LCGGenerator(
            modulus=2**32,
            multiplier=214013,
            increment=2531011,
            seed=seed
        )
    
# Legacy alias for backward compatibility
LCG = LCGGenerator
