#!/usr/bin/env python
"""
Permuted Congruential Generator for HDC

HDC-compatible wrapper for PCG random number generation.
"""

import random
from typing import Any, Dict, List, Optional

from pyhdc.generation.base import HDCGenerator


class PCGGenerator(HDCGenerator):
    """
    Permuted Congruential Generator for hypervector generation.
    
    Uses an LCG with a permutation function for improved statistical properties.
    """
    
    def __init__(
        self,
        state_bits: int = 64,
        output_bits: int = 32,
        multiplier: int = 6364136223846793005,
        increment: int = 1442695040888963407,
        seed: Optional[int] = None,
        permutation: str = "xsh-rs"
    ) -> None:
        """
        Initialize PCG generator.
        
        Args:
            state_bits: Number of bits in internal state
            output_bits: Number of bits in output
            multiplier: LCG multiplier value
            increment: LCG increment value
            seed: Optional seed for reproducibility
            permutation: Output permutation method ("xsh-rs", "xsh-rr", "rxs-m-xs")
        """
        self._state_bits = state_bits
        self._output_bits = output_bits
        self._multiplier = multiplier
        self._increment = increment
        self._permutation = permutation
        self._validate_parameters()
        super().__init__(seed)
    
    def _validate_parameters(self) -> None:
        """Validate PCG parameters."""
        if not isinstance(self._state_bits, int) or self._state_bits <= 0:
            raise ValueError("State bits must be a positive integer")
        
        if not isinstance(self._output_bits, int) or self._output_bits <= 0:
            raise ValueError("Output bits must be a positive integer")
        
        if self._output_bits > self._state_bits:
            raise ValueError(f"Output bits must not exceed state bits ({self._state_bits})")
        
        if not isinstance(self._multiplier, int) or self._multiplier <= 0 or self._multiplier % 2 == 0:
            raise ValueError("Multiplier must be a positive odd integer")
        
        if not isinstance(self._increment, int) or self._increment < 0:
            raise ValueError("Increment must be a non-negative integer")
        
        if self._permutation not in ["xsh-rs", "xsh-rr", "rxs-m-xs"]:
            raise ValueError("Permutation must be one of: xsh-rs, xsh-rr, rxs-m-xs")
    
    def _configure_internal(self) -> None:
        """Configure the PCG state."""
        self._state_mask = (1 << self._state_bits) - 1
        self._output_mask = (1 << self._output_bits) - 1
        
        if self._seed is None:
            self._state = random.randint(0, self._state_mask)
        else:
            if not isinstance(self._seed, int) or self._seed < 0 or self._seed > self._state_mask:
                raise ValueError(f"Seed must be between 0 and {self._state_mask}")
            self._state = self._seed
        
        self._initial_state = self._state
    
    def _lcg_step(self) -> None:
        """Perform one LCG step to advance the internal state."""
        self._state = ((self._state * self._multiplier) + self._increment) & self._state_mask
    
    def _permute_xsh_rs(self, state: int) -> int:
        """XOR-shift-rotate-shift permutation."""
        xorshifted = ((state >> ((self._state_bits + self._output_bits) // 2)) ^ state) & self._output_mask
        rotation = state >> (self._state_bits - 5)
        rotation &= (self._output_bits - 1)
        return ((xorshifted >> rotation) | (xorshifted << (self._output_bits - rotation))) & self._output_mask
    
    def _permute_xsh_rr(self, state: int) -> int:
        """XOR-shift-rotate-rotate permutation."""
        xorshifted = ((state >> ((self._state_bits + self._output_bits) // 2)) ^ state) & self._output_mask
        rotation1 = state >> (self._state_bits - 4)
        rotation1 &= (self._output_bits - 1)
        rotated1 = ((xorshifted >> rotation1) | (xorshifted << (self._output_bits - rotation1))) & self._output_mask
        rotation2 = (state >> (self._state_bits - 8)) & 3
        return ((rotated1 >> rotation2) | (rotated1 << (self._output_bits - rotation2))) & self._output_mask
    
    def _permute_rxs_m_xs(self, state: int) -> int:
        """Rotate-XOR-shift, multiply, XOR-shift permutation."""
        rotation = state >> (self._state_bits - 5)
        rotation &= (self._output_bits - 1)
        rotated = ((state >> rotation) | (state << (self._state_bits - rotation))) & self._state_mask
        xorshifted = (rotated ^ (rotated >> ((self._state_bits + self._output_bits) // 2))) & self._output_mask
        multiplier = 0xAEF17502108EF2D9 & self._output_mask
        multiplied = (xorshifted * multiplier) & self._output_mask
        return (multiplied ^ (multiplied >> (self._output_bits // 2))) & self._output_mask
    
    def _permute_output(self, state: int) -> int:
        """Apply the selected permutation function."""
        if self._permutation == "xsh-rs":
            return self._permute_xsh_rs(state)
        elif self._permutation == "xsh-rr":
            return self._permute_xsh_rr(state)
        elif self._permutation == "rxs-m-xs":
            return self._permute_rxs_m_xs(state)
        else:
            raise ValueError(f"Unknown permutation: {self._permutation}")
    
    def _next_value(self) -> int:
        """Generate next value."""
        old_state = self._state
        self._lcg_step()
        return self._permute_output(old_state)
    
    def _next_bit(self) -> int:
        """Generate next bit."""
        return self._next_value() & 1
    
    def _next_word(self, word_size: int = 32) -> int:
        """Generate next word."""
        if word_size <= self._output_bits:
            return self._next_value() & ((1 << word_size) - 1)
        else:
            # Combine multiple outputs
            result = 0
            bits_collected = 0
            while bits_collected < word_size:
                value = self._next_value()
                result |= (value << bits_collected)
                bits_collected += self._output_bits
            return result & ((1 << word_size) - 1)
    
    def generate_floats(self, length: int, min_val: float = -1.0, 
                       max_val: float = 1.0) -> List[float]:
        """Generate floats efficiently using PCG."""
        result = []
        max_output = 1 << self._output_bits
        for _ in range(length):
            value = self._next_value()
            normalized = value / max_output
            result.append(min_val + normalized * (max_val - min_val))
        return result
    
    def set_parameters(
        self,
        state_bits: Optional[int] = None,
        output_bits: Optional[int] = None,
        multiplier: Optional[int] = None,
        increment: Optional[int] = None,
        permutation: Optional[str] = None,
        seed: Optional[int] = None
    ) -> None:
        """
        Set PCG parameters.
        
        Args:
            state_bits: Number of bits in internal state
            output_bits: Number of bits in output
            multiplier: LCG multiplier value
            increment: LCG increment value
            permutation: Output permutation method
            seed: The seed value
        """
        if state_bits is not None:
            self._state_bits = state_bits
        
        if output_bits is not None:
            self._output_bits = output_bits
        
        if multiplier is not None:
            self._multiplier = multiplier
        
        if increment is not None:
            self._increment = increment
        
        if permutation is not None:
            self._permutation = permutation
        
        if seed is not None:
            self._seed = seed
        
        self._validate_parameters()
        self._configure_internal()
    
    def set_state_bits(self, state_bits: int) -> None:
        """Set the state bits parameter."""
        self.set_parameters(state_bits=state_bits)
    
    def set_output_bits(self, output_bits: int) -> None:
        """Set the output bits parameter."""
        self.set_parameters(output_bits=output_bits)
    
    def set_multiplier(self, multiplier: int) -> None:
        """Set the multiplier parameter."""
        self.set_parameters(multiplier=multiplier)
    
    def set_increment(self, increment: int) -> None:
        """Set the increment parameter."""
        self.set_parameters(increment=increment)
    
    def set_permutation(self, permutation: str) -> None:
        """Set the permutation method."""
        self.set_parameters(permutation=permutation)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current PCG parameters."""
        return {
            "state_bits": self._state_bits,
            "output_bits": self._output_bits,
            "multiplier": self._multiplier,
            "increment": self._increment,
            "permutation": self._permutation,
            "seed": self._seed
        }
    
    def get_state_bits(self) -> int:
        """Get the state bits parameter."""
        return self._state_bits
    
    def get_output_bits(self) -> int:
        """Get the output bits parameter."""
        return self._output_bits
    
    def get_multiplier(self) -> int:
        """Get the multiplier parameter."""
        return self._multiplier
    
    def get_increment(self) -> int:
        """Get the increment parameter."""
        return self._increment
    
    def get_permutation(self) -> str:
        """Get the permutation method."""
        return self._permutation
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._state = self._initial_state
    
    def get_state(self) -> int:
        """Get current state."""
        return self._state


class MultiplicativePCGGenerator(PCGGenerator):
    """
    PCG based on Multiplicative Congruential Generator.
    
    Uses MCG (increment = 0) instead of LCG for the underlying generator.
    """
    
    def __init__(
        self,
        state_bits: int = 64,
        output_bits: int = 32,
        multiplier: int = 6364136223846793005,
        seed: Optional[int] = None
    ) -> None:
        """Initialize multiplicative PCG."""
        super().__init__(state_bits, output_bits, multiplier, 0, seed, "xsh-rs")
    
    def _configure_internal(self) -> None:
        """Configure with non-zero seed."""
        super()._configure_internal()
        if self._state == 0:
            self._state = 1
            self._initial_state = 1
    
    def set_parameters(
        self,
        state_bits: Optional[int] = None,
        output_bits: Optional[int] = None,
        multiplier: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> None:
        """Set parameters (increment always 0)."""
        if "increment" in kwargs and kwargs["increment"] != 0:
            raise ValueError("Multiplicative PCG always has increment = 0")
        super().set_parameters(state_bits, output_bits, multiplier, 0, None, seed)


# Predefined PCG configurations
class CommonPCGGenerators:
    """Factory for common PCG configurations."""
    
    @staticmethod
    def pcg32(seed: Optional[int] = None) -> PCGGenerator:
        """Standard 32-bit output PCG with 64-bit state."""
        return PCGGenerator(
            state_bits=64,
            output_bits=32,
            multiplier=6364136223846793005,
            increment=1442695040888963407,
            seed=seed,
            permutation="xsh-rs"
        )
    
    @staticmethod
    def pcg64(seed: Optional[int] = None) -> PCGGenerator:
        """64-bit output PCG with 64-bit state."""
        return PCGGenerator(
            state_bits=64,
            output_bits=64,
            multiplier=6364136223846793005,
            increment=1442695040888963407,
            seed=seed,
            permutation="xsh-rr"
        )
    
    @staticmethod
    def pcg16(seed: Optional[int] = None) -> PCGGenerator:
        """16-bit output PCG with 32-bit state."""
        return PCGGenerator(
            state_bits=32,
            output_bits=16,
            multiplier=747796405,
            increment=2891336453,
            seed=seed,
            permutation="xsh-rs"
        )
    
    @staticmethod
    def pcg_fast(seed: Optional[int] = None) -> MultiplicativePCGGenerator:
        """Fast MCG-based PCG."""
        return MultiplicativePCGGenerator(
            state_bits=64,
            output_bits=32,
            multiplier=6364136223846793005,
            seed=seed
        )


# Legacy alias for backward compatibility
PCG = PCGGenerator


class XSHRS_PCGGenerator(PCGGenerator):
    """
    Permuted Congruential Generator for hypervector generation, using the xsh-rs permutation method.
    
    Uses an LCG with a permutation function for improved statistical properties.
    """
    
    def __init__(
        self,
        state_bits: int = 64,
        output_bits: int = 32,
        multiplier: int = 6364136223846793005,
        increment: int = 1442695040888963407,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize PCG generator.
        
        Args:
            state_bits: Number of bits in internal state
            output_bits: Number of bits in output
            multiplier: LCG multiplier value
            increment: LCG increment value
            seed: Optional seed for reproducibility
        """
        super().__init__(state_bits, output_bits, multiplier, increment, seed, "xsh-rs")


class XSHRR_PCGGenerator(PCGGenerator):
    """
    Permuted Congruential Generator for hypervector generation, using the xsh-rr permutation method.
    
    Uses an LCG with a permutation function for improved statistical properties.
    """
    
    def __init__(
        self,
        state_bits: int = 64,
        output_bits: int = 32,
        multiplier: int = 6364136223846793005,
        increment: int = 1442695040888963407,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize PCG generator.
        
        Args:
            state_bits: Number of bits in internal state
            output_bits: Number of bits in output
            multiplier: LCG multiplier value
            increment: LCG increment value
            seed: Optional seed for reproducibility
        """
        super().__init__(state_bits, output_bits, multiplier, increment, seed, "xsh-rr")


class RXS_M_XS_PCGGenerator(HDCGenerator):
    """
    Permuted Congruential Generator for hypervector generation, using the rxs-m-xs permutation method.
    
    Uses an LCG with a permutation function for improved statistical properties.
    """
    
    def __init__(
        self,
        state_bits: int = 64,
        output_bits: int = 32,
        multiplier: int = 6364136223846793005,
        increment: int = 1442695040888963407,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize PCG generator.
        
        Args:
            state_bits: Number of bits in internal state
            output_bits: Number of bits in output
            multiplier: LCG multiplier value
            increment: LCG increment value
            seed: Optional seed for reproducibility
        """
        super().__init__(state_bits, output_bits, multiplier, increment, seed, "rxs-m-xs")
