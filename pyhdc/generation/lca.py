#!/usr/bin/env python
"""
Linear Cellular Automata for HDC

HDC-compatible wrapper for LCA random number generation.
"""

import random
from typing import Any, Dict, List, Optional

from pyhdc.generation.base import HDCGenerator


class LCAGenerator(HDCGenerator):
    """
    Linear Cellular Automata for hypervector generation.
    
    Uses cellular automata evolution to generate pseudorandom sequences.
    """
    
    def __init__(
        self,
        width: int = 32,
        rule: int = 30,
        boundary: str = "periodic",
        seed: Optional[int] = None,
        extraction_method: str = "lsb"
    ) -> None:
        """
        Initialize LCA generator.
        
        Args:
            width: The width of the cellular automaton in cells
            rule: The rule number defining the local update function
            boundary: Boundary condition ("periodic", "fixed", "null")
            seed: Optional seed for reproducibility
            extraction_method: Method to extract bits ("lsb", "msb", "center", "parity")
        """
        self._width = width
        self._rule = rule
        self._boundary = boundary
        self._extraction_method = extraction_method
        self._validate_parameters()
        super().__init__(seed)
    
    def _validate_parameters(self) -> None:
        """Validate LCA parameters."""
        if not isinstance(self._width, int) or self._width <= 0:
            raise ValueError("Width must be a positive integer")
        
        if not isinstance(self._rule, int) or self._rule < 0:
            raise ValueError("Rule must be a non-negative integer")
        
        if self._boundary not in ["periodic", "fixed", "null"]:
            raise ValueError("Boundary must be one of: periodic, fixed, null")
        
        if self._extraction_method not in ["lsb", "msb", "center", "parity"]:
            raise ValueError("Extraction method must be one of: lsb, msb, center, parity")
    
    def _configure_internal(self) -> None:
        """Configure the LCA state."""
        self._max_value = (1 << self._width) - 1
        self._rule_table = self._build_rule_table(self._rule)
        
        if self._seed is None:
            self._state = random.randint(1, self._max_value)
        else:
            if not isinstance(self._seed, int) or self._seed < 0 or self._seed > self._max_value:
                raise ValueError(f"Seed must be between 0 and {self._max_value}")
            self._state = self._seed
        
        self._initial_state = self._state
    
    def _build_rule_table(self, rule: int) -> List[int]:
        """Build lookup table for the cellular automaton rule."""
        table = []
        for i in range(8):
            table.append((rule >> i) & 1)
        return table
    
    def _get_cell(self, position: int) -> int:
        """Get the value of a cell at given position with boundary handling."""
        if 0 <= position < self._width:
            return (self._state >> position) & 1
        
        if self._boundary == "periodic":
            return (self._state >> (position % self._width)) & 1
        else:  # fixed or null boundary
            return 0
    
    def _get_neighborhood(self, position: int) -> int:
        """Get the neighborhood value for a cell position."""
        left = self._get_cell(position - 1)
        center = self._get_cell(position)
        right = self._get_cell(position + 1)
        
        return (left << 2) | (center << 1) | right
    
    def _next_generation(self) -> int:
        """Compute the next generation of the cellular automaton."""
        new_state = 0
        
        for i in range(self._width):
            neighborhood = self._get_neighborhood(i)
            new_bit = self._rule_table[neighborhood]
            if new_bit:
                new_state |= (1 << i)
        
        self._state = new_state
        return self._state
    
    def _extract_bit(self, state: int) -> int:
        """Extract a bit from state using the configured method."""
        if self._extraction_method == "lsb":
            return state & 1
        elif self._extraction_method == "msb":
            return (state >> (self._width - 1)) & 1
        elif self._extraction_method == "center":
            return (state >> (self._width // 2)) & 1
        else:  # parity
            return bin(state).count('1') % 2
    
    def _next_bit(self) -> int:
        """Generate next bit."""
        state = self._next_generation()
        return self._extract_bit(state)
    
    def _next_word(self, word_size: int = 32) -> int:
        """Generate next word by evolving CA and extracting bits."""
        word = 0
        for i in range(word_size):
            word |= (self._next_bit() << i)
        return word
    
    def set_parameters(
        self,
        width: Optional[int] = None,
        rule: Optional[int] = None,
        boundary: Optional[str] = None,
        extraction_method: Optional[str] = None,
        seed: Optional[int] = None
    ) -> None:
        """
        Set LCA parameters.
        
        Args:
            width: The width of the cellular automaton
            rule: The rule number
            boundary: Boundary condition
            extraction_method: Bit extraction method
            seed: The seed value
        """
        if width is not None:
            self._width = width
        
        if rule is not None:
            self._rule = rule
        
        if boundary is not None:
            self._boundary = boundary
        
        if extraction_method is not None:
            self._extraction_method = extraction_method
        
        if seed is not None:
            self._seed = seed
        
        self._validate_parameters()
        self._configure_internal()
    
    def set_width(self, width: int) -> None:
        """Set the width parameter."""
        self.set_parameters(width=width)
    
    def set_rule(self, rule: int) -> None:
        """Set the rule number."""
        self.set_parameters(rule=rule)
    
    def set_boundary(self, boundary: str) -> None:
        """Set the boundary condition."""
        self.set_parameters(boundary=boundary)
    
    def set_extraction_method(self, extraction_method: str) -> None:
        """Set the bit extraction method."""
        self.set_parameters(extraction_method=extraction_method)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current LCA parameters."""
        return {
            "width": self._width,
            "rule": self._rule,
            "boundary": self._boundary,
            "extraction_method": self._extraction_method,
            "seed": self._seed
        }
    
    def get_width(self) -> int:
        """Get the width parameter."""
        return self._width
    
    def get_rule(self) -> int:
        """Get the rule number."""
        return self._rule
    
    def get_boundary(self) -> str:
        """Get the boundary condition."""
        return self._boundary
    
    def get_extraction_method(self) -> str:
        """Get the extraction method."""
        return self._extraction_method
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._state = self._initial_state
    
    def get_state(self) -> int:
        """Get current state."""
        return self._state
    
    def get_state_as_array(self) -> List[int]:
        """Get current state as an array of cell values."""
        return [(self._state >> i) & 1 for i in range(self._width)]


class ElementaryLCAGenerator(LCAGenerator):
    """Elementary LCA generator using standard elementary CA rules."""
    
    def __init__(
        self,
        width: int = 32,
        rule: int = 30,
        boundary: str = "periodic",
        seed: Optional[int] = None
    ) -> None:
        """Initialize elementary LCA."""
        super().__init__(width, rule, boundary, seed, "lsb")


class TotalisticLCAGenerator(LCAGenerator):
    """
    Totalistic LCA generator.
    
    Uses totalistic rules where next state depends only on the sum
    of neighborhood values.
    """
    
    def __init__(
        self,
        width: int = 32,
        rule: int = 14,
        boundary: str = "periodic",
        seed: Optional[int] = None
    ) -> None:
        """Initialize totalistic LCA."""
        super().__init__(width, rule, boundary, seed, "lsb")
    
    def _build_rule_table(self, rule: int) -> List[int]:
        """Build lookup table for totalistic CA rules."""
        # For 3-cell totalistic neighborhood, sums can be 0, 1, 2, 3
        table = []
        for i in range(4):
            table.append((rule >> i) & 1)
        return table
    
    def _get_neighborhood_sum(self, position: int) -> int:
        """Get the sum of the neighborhood for a cell position."""
        left = self._get_cell(position - 1)
        center = self._get_cell(position)
        right = self._get_cell(position + 1)
        return left + center + right
    
    def _next_generation(self) -> int:
        """Compute next generation using totalistic rules."""
        new_state = 0
        
        for i in range(self._width):
            neighborhood_sum = self._get_neighborhood_sum(i)
            new_bit = self._rule_table[neighborhood_sum]
            if new_bit:
                new_state |= (1 << i)
        
        self._state = new_state
        return self._state


# Predefined LCA configurations
class CommonLCAGenerators:
    """Factory for common LCA configurations."""
    
    @staticmethod
    def rule_30(width: int = 64, seed: Optional[int] = None) -> ElementaryLCAGenerator:
        """Rule 30 - chaotic behavior, used in Mathematica's RandomInteger."""
        return ElementaryLCAGenerator(
            width=width,
            rule=30,
            boundary="periodic",
            seed=seed
        )
    
    @staticmethod
    def rule_90(width: int = 64, seed: Optional[int] = None) -> ElementaryLCAGenerator:
        """Rule 90 - simple fractal patterns, XOR rule."""
        return ElementaryLCAGenerator(
            width=width,
            rule=90,
            boundary="periodic",
            seed=seed
        )
    
    @staticmethod
    def rule_110(width: int = 64, seed: Optional[int] = None) -> ElementaryLCAGenerator:
        """Rule 110 - universal computation."""
        return ElementaryLCAGenerator(
            width=width,
            rule=110,
            boundary="periodic",
            seed=seed
        )
    
    @staticmethod
    def rule_150(width: int = 64, seed: Optional[int] = None) -> ElementaryLCAGenerator:
        """Rule 150 - another XOR rule with good randomness."""
        return ElementaryLCAGenerator(
            width=width,
            rule=150,
            boundary="periodic",
            seed=seed
        )
    
    @staticmethod
    def majority_totalistic(width: int = 64, seed: Optional[int] = None) -> TotalisticLCAGenerator:
        """Totalistic majority rule."""
        return TotalisticLCAGenerator(
            width=width,
            rule=14,  # Binary: 1110 - majority rule
            boundary="periodic",
            seed=seed
        )


# Legacy alias for backward compatibility
LCA = ElementaryLCAGenerator
