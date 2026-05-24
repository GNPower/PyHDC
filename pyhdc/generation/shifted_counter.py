#!/usr/bin/env python
"""
Shifted Counter Generators for HDC

HDC-compatible wrapper for counter-based random number generation with
invertible mappings (Feistel, SPN, ARX, etc.).
"""

import random
from typing import Any, Callable, Dict, List, Optional

from pyhdc.generation.base import HDCGenerator


class ShiftedCounterGenerator(HDCGenerator):
    """
    Base class for shifted counter generators with invertible mappings.
    
    Combines a simple counter with cryptographic-style permutations
    for pseudorandom number generation.
    """
    
    def __init__(
        self,
        bit_width: int = 32,
        shift_amount: int = 13,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize shifted counter generator.
        
        Args:
            bit_width: Width of the counter in bits
            shift_amount: Amount to shift the counter
            seed: Optional seed for reproducibility
        """
        self._bit_width = bit_width
        self._shift_amount = shift_amount % bit_width if bit_width > 0 else 0
        self._validate_parameters()
        super().__init__(seed)
    
    def _validate_parameters(self) -> None:
        """Validate counter parameters."""
        if not isinstance(self._bit_width, int) or self._bit_width <= 0:
            raise ValueError("Bit width must be a positive integer")
        
        if not isinstance(self._shift_amount, int) or self._shift_amount < 0:
            raise ValueError("Shift amount must be a non-negative integer")
    
    def _configure_internal(self) -> None:
        """Configure the counter state."""
        self._max_value = (1 << self._bit_width) - 1
        
        if self._seed is None:
            self._counter = random.randint(0, self._max_value)
        else:
            if not isinstance(self._seed, int) or self._seed < 0 or self._seed > self._max_value:
                raise ValueError(f"Seed must be between 0 and {self._max_value}")
            self._counter = self._seed
        
        self._initial_counter = self._counter
    
    def _shift_counter(self) -> int:
        """Apply shift operation to current counter value."""
        shifted = ((self._counter << self._shift_amount) | 
                  (self._counter >> (self._bit_width - self._shift_amount))) & self._max_value
        return shifted
    
    def _apply_mapping(self, value: int) -> int:
        """Apply invertible mapping (implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _apply_mapping")
    
    def _next_value(self) -> int:
        """Generate next value."""
        shifted_value = self._shift_counter()
        output = self._apply_mapping(shifted_value)
        self._counter = (self._counter + 1) & self._max_value
        return output
    
    def _next_bit(self) -> int:
        """Generate next bit."""
        return self._next_value() & 1
    
    def _next_word(self, word_size: int = 32) -> int:
        """Generate next word."""
        if word_size <= self._bit_width:
            return self._next_value() & ((1 << word_size) - 1)
        else:
            # Combine multiple outputs
            result = 0
            bits_collected = 0
            while bits_collected < word_size:
                value = self._next_value()
                result |= (value << bits_collected)
                bits_collected += self._bit_width
            return result & ((1 << word_size) - 1)
    
    def set_parameters(
        self,
        bit_width: Optional[int] = None,
        shift_amount: Optional[int] = None,
        seed: Optional[int] = None
    ) -> None:
        """Set counter parameters."""
        if bit_width is not None:
            self._bit_width = bit_width
            self._shift_amount = self._shift_amount % bit_width if bit_width > 0 else 0
        
        if shift_amount is not None:
            self._shift_amount = shift_amount % self._bit_width if self._bit_width > 0 else 0
        
        if seed is not None:
            self._seed = seed
        
        self._validate_parameters()
        self._configure_internal()
    
    def set_bit_width(self, bit_width: int) -> None:
        """Set the bit width parameter."""
        self.set_parameters(bit_width=bit_width)
    
    def set_shift_amount(self, shift_amount: int) -> None:
        """Set the shift amount parameter."""
        self.set_parameters(shift_amount=shift_amount)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {
            "bit_width": self._bit_width,
            "shift_amount": self._shift_amount,
            "seed": self._seed
        }
    
    def get_bit_width(self) -> int:
        """Get the bit width parameter."""
        return self._bit_width
    
    def get_shift_amount(self) -> int:
        """Get the shift amount parameter."""
        return self._shift_amount
    
    def get_period(self) -> int:
        """Get the period length."""
        return 1 << self._bit_width
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._counter = self._initial_counter
    
    def get_state(self) -> int:
        """Get current counter state."""
        return self._counter


class FeistelCounterGenerator(ShiftedCounterGenerator):
    """
    Shifted counter with Feistel network mapping.
    
    Uses a Feistel network to create an invertible permutation.
    """
    
    def __init__(
        self,
        bit_width: int = 32,
        shift_amount: int = 13,
        rounds: int = 4,
        round_keys: Optional[List[int]] = None,
        seed: Optional[int] = None
    ) -> None:
        """Initialize Feistel counter."""
        if bit_width % 2 != 0:
            raise ValueError("Bit width must be even for Feistel networks")
        
        if not isinstance(rounds, int) or rounds <= 0:
            raise ValueError("Rounds must be a positive integer")
        
        self._rounds = rounds
        self._round_keys = round_keys
        super().__init__(bit_width, shift_amount, seed)
    
    def _configure_internal(self) -> None:
        """Configure with Feistel-specific parameters."""
        super()._configure_internal()

        self._half_width = self._bit_width // 2
        self._half_mask = (1 << self._half_width) - 1

        if self._round_keys is None:
            # Use a seeded local RNG so round keys are deterministic when a seed is set
            rng = random.Random(self._seed)
            self._round_keys = [rng.randint(0, self._half_mask) for _ in range(self._rounds)]
        else:
            if len(self._round_keys) != self._rounds:
                raise ValueError(f"Must provide {self._rounds} round keys")
            self._round_keys = [k & self._half_mask for k in self._round_keys]
    
    def _feistel_function(self, half_block: int, round_key: int) -> int:
        """Feistel function (F-function)."""
        result = half_block ^ round_key
        
        # Add non-linearity with rotations and XOR
        result = ((result << 3) | (result >> (self._half_width - 3))) & self._half_mask
        result ^= (result >> 1)
        result = ((result << 1) | (result >> (self._half_width - 1))) & self._half_mask
        
        return result
    
    def _apply_mapping(self, value: int) -> int:
        """Apply Feistel network mapping."""
        left = (value >> self._half_width) & self._half_mask
        right = value & self._half_mask
        
        for i in range(self._rounds):
            new_left = right
            new_right = left ^ self._feistel_function(right, self._round_keys[i])
            new_right &= self._half_mask
            left, right = new_left, new_right
        
        return ((left << self._half_width) | right) & self._max_value
    
    def set_round_keys(self, round_keys: List[int]) -> None:
        """Set the round keys."""
        if len(round_keys) != self._rounds:
            raise ValueError(f"Must provide {self._rounds} round keys")
        self._round_keys = [k & self._half_mask for k in round_keys]
    
    def get_round_keys(self) -> List[int]:
        """Get the round keys."""
        return self._round_keys.copy()


class ARXCounterGenerator(ShiftedCounterGenerator):
    """
    Shifted counter with Addition-Rotation-XOR mapping.
    
    Uses ARX operations for fast, secure mixing.
    """
    
    def __init__(
        self,
        bit_width: int = 32,
        shift_amount: int = 11,
        constants: Optional[List[int]] = None,
        rotations: Optional[List[int]] = None,
        seed: Optional[int] = None
    ) -> None:
        """Initialize ARX counter."""
        self._constants = constants
        self._rotations = rotations
        super().__init__(bit_width, shift_amount, seed)
    
    def _configure_internal(self) -> None:
        """Configure with ARX-specific parameters."""
        super()._configure_internal()
        
        if self._constants is None:
            # Default constants (similar to ChaCha/Salsa)
            self._constants = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]
            self._constants = [c & self._max_value for c in self._constants]
        else:
            self._constants = [c & self._max_value for c in self._constants]
        
        if self._rotations is None:
            self._rotations = [7, 12, 17, 22] if self._bit_width >= 32 else [3, 5, 7, 11]
    
    def _rotleft(self, value: int, amount: int) -> int:
        """Rotate value left."""
        amount %= self._bit_width
        return ((value << amount) | (value >> (self._bit_width - amount))) & self._max_value
    
    def _apply_mapping(self, value: int) -> int:
        """Apply ARX mapping."""
        result = value
        
        for i, (constant, rotation) in enumerate(zip(self._constants, self._rotations)):
            # Addition
            result = (result + constant) & self._max_value
            # Rotation
            result = self._rotleft(result, rotation)
            # XOR
            xor_val = (constant << (i + 1)) & self._max_value
            result ^= xor_val
        
        return result
    
    def set_constants(self, constants: List[int]) -> None:
        """Set the ARX constants."""
        self._constants = [c & self._max_value for c in constants]
    
    def set_rotations(self, rotations: List[int]) -> None:
        """Set the rotation amounts."""
        self._rotations = rotations
    
    def get_constants(self) -> List[int]:
        """Get the ARX constants."""
        return self._constants.copy()
    
    def get_rotations(self) -> List[int]:
        """Get the rotation amounts."""
        return self._rotations.copy()


class SPNCounterGenerator(ShiftedCounterGenerator):
    """
    Shifted counter with Substitution-Permutation Network mapping.
    
    Uses S-boxes and permutation for invertible mapping.
    """
    
    def __init__(
        self,
        bit_width: int = 32,
        shift_amount: int = 17,
        sbox_size: int = 4,
        sbox: Optional[List[int]] = None,
        pbox: Optional[List[int]] = None,
        seed: Optional[int] = None
    ) -> None:
        """Initialize SPN counter."""
        if bit_width % sbox_size != 0:
            raise ValueError(f"Bit width must be divisible by S-box size ({sbox_size})")
        
        self._sbox_size = sbox_size
        self._sbox = sbox
        self._pbox = pbox
        super().__init__(bit_width, shift_amount, seed)
    
    def _configure_internal(self) -> None:
        """Configure with SPN-specific parameters."""
        super()._configure_internal()
        
        self._sbox_count = self._bit_width // self._sbox_size
        self._sbox_mask = (1 << self._sbox_size) - 1
        
        if self._sbox is None:
            # Generate random permutation for S-box
            sbox_values = list(range(1 << self._sbox_size))
            random.shuffle(sbox_values)
            self._sbox = sbox_values
        else:
            if len(self._sbox) != (1 << self._sbox_size):
                raise ValueError(f"S-box must have {1 << self._sbox_size} entries")
            if sorted(self._sbox) != list(range(1 << self._sbox_size)):
                raise ValueError("S-box must be a valid permutation")
        
        if self._pbox is None:
            # Generate random bit permutation
            self._pbox = list(range(self._bit_width))
            random.shuffle(self._pbox)
        else:
            if len(self._pbox) != self._bit_width or sorted(self._pbox) != list(range(self._bit_width)):
                raise ValueError("P-box must be a valid bit permutation")
    
    def _apply_sbox(self, value: int) -> int:
        """Apply S-box substitution."""
        result = 0
        for i in range(self._sbox_count):
            sbox_input = (value >> (i * self._sbox_size)) & self._sbox_mask
            sbox_output = self._sbox[sbox_input]
            result |= (sbox_output << (i * self._sbox_size))
        return result & self._max_value
    
    def _apply_pbox(self, value: int) -> int:
        """Apply P-box permutation."""
        result = 0
        for i in range(self._bit_width):
            if (value >> i) & 1:
                result |= (1 << self._pbox[i])
        return result
    
    def _apply_mapping(self, value: int) -> int:
        """Apply SPN mapping."""
        substituted = self._apply_sbox(value)
        permuted = self._apply_pbox(substituted)
        return permuted
    
    def set_sbox(self, sbox: List[int]) -> None:
        """Set the S-box."""
        if len(sbox) != (1 << self._sbox_size):
            raise ValueError(f"S-box must have {1 << self._sbox_size} entries")
        if sorted(sbox) != list(range(1 << self._sbox_size)):
            raise ValueError("S-box must be a valid permutation")
        self._sbox = sbox.copy()
    
    def set_pbox(self, pbox: List[int]) -> None:
        """Set the P-box."""
        if len(pbox) != self._bit_width or sorted(pbox) != list(range(self._bit_width)):
            raise ValueError("P-box must be a valid bit permutation")
        self._pbox = pbox.copy()
    
    def get_sbox(self) -> List[int]:
        """Get the S-box."""
        return self._sbox.copy()
    
    def get_pbox(self) -> List[int]:
        """Get the P-box."""
        return self._pbox.copy()


class CustomMappingCounterGenerator(ShiftedCounterGenerator):
    """
    Shifted counter with custom user-defined mapping.
    
    Allows users to provide their own invertible mapping function.
    """
    
    def __init__(
        self,
        bit_width: int = 32,
        shift_amount: int = 15,
        mapping_func: Optional[Callable[[int], int]] = None,
        seed: Optional[int] = None
    ) -> None:
        """Initialize custom mapping counter."""
        self._mapping_func = mapping_func
        super().__init__(bit_width, shift_amount, seed)
    
    def _configure_internal(self) -> None:
        """Configure with default mapping if none provided."""
        super()._configure_internal()
        
        if self._mapping_func is None:
            # Default simple hash function
            def simple_hash(x):
                x ^= x >> 16
                x *= 0x85ebca6b
                x ^= x >> 13
                x *= 0xc2b2ae35
                x ^= x >> 16
                return x
            self._mapping_func = simple_hash
    
    def _apply_mapping(self, value: int) -> int:
        """Apply custom mapping."""
        return self._mapping_func(value) & self._max_value
    
    def set_mapping_function(self, mapping_func: Callable[[int], int]) -> None:
        """Set a new mapping function."""
        self._mapping_func = mapping_func


# Predefined counter configurations
class CommonCounterGenerators:
    """Factory for common counter configurations."""
    
    @staticmethod
    def feistel_32bit(seed: Optional[int] = None) -> FeistelCounterGenerator:
        """32-bit Feistel counter with standard parameters."""
        return FeistelCounterGenerator(
            bit_width=32,
            shift_amount=13,
            rounds=4,
            seed=seed
        )
    
    @staticmethod
    def feistel_64bit(seed: Optional[int] = None) -> FeistelCounterGenerator:
        """64-bit Feistel counter with standard parameters."""
        return FeistelCounterGenerator(
            bit_width=64,
            shift_amount=21,
            rounds=6,
            seed=seed
        )
    
    @staticmethod
    def arx_32bit(seed: Optional[int] = None) -> ARXCounterGenerator:
        """32-bit ARX counter."""
        return ARXCounterGenerator(
            bit_width=32,
            shift_amount=11,
            seed=seed
        )
    
    @staticmethod
    def arx_64bit(seed: Optional[int] = None) -> ARXCounterGenerator:
        """64-bit ARX counter."""
        return ARXCounterGenerator(
            bit_width=64,
            shift_amount=17,
            seed=seed
        )
    
    @staticmethod
    def spn_32bit(seed: Optional[int] = None) -> SPNCounterGenerator:
        """32-bit SPN counter with 4-bit S-boxes."""
        return SPNCounterGenerator(
            bit_width=32,
            shift_amount=17,
            sbox_size=4,
            seed=seed
        )
    
    @staticmethod
    def simple_hash(bit_width: int = 32, seed: Optional[int] = None) -> CustomMappingCounterGenerator:
        """Counter with simple hash-based mapping."""
        return CustomMappingCounterGenerator(
            bit_width=bit_width,
            shift_amount=15,
            seed=seed
        )
    

# Legacy alias for backward compatibility
ShiftedCounter = FeistelCounterGenerator
