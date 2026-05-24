import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class HDCGenerator(ABC):
    """
    Abstract base class for HDC-compatible random number generators.
    
    Generators can produce bits, words, or floats, and are used to generate
    hypervectors with specific properties (e.g., using LFSRs, LCGs, etc.).
    """
    
    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the generator.
        
        Args:
            seed: Optional seed for reproducibility
        """
        self._seed = seed
        self._state: Any = None
        self._configure_internal()
    
    @abstractmethod
    def _configure_internal(self) -> None:
        """
        Internal configuration called during initialization.
        Should set up the generator state based on parameters.
        """
    
    @abstractmethod
    def _next_bit(self) -> int:
        """
        Generate the next bit (0 or 1).
        
        Returns:
            The next bit (0 or 1)
            
        Raises:
            NotImplementedError: If the generator doesn't support bit generation
        """
    
    @abstractmethod
    def _next_word(self, word_size: int = 32) -> int:
        """
        Generate the next word of specified bit width.
        
        Args:
            word_size: Number of bits in the word
            
        Returns:
            The next word as an integer
            
        Raises:
            NotImplementedError: If the generator doesn't support word generation
        """
    
    def generate_bits(self, length: int) -> List[int]:
        """
        Generate a sequence of bits.
        
        Args:
            length: Number of bits to generate
            
        Returns:
            List of bits (0s and 1s)
            
        Raises:
            NotImplementedError: If the generator doesn't support bit generation
            ValueError: If length is not positive
        """
        if not isinstance(length, int) or length <= 0:
            raise ValueError("Length must be a positive integer.")
        
        try:
            return [self._next_bit() for _ in range(length)]
        except NotImplementedError:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support bit generation. "
                f"Use a generator that implements _next_bit()."
            )
    
    def generate_words(self, length: int, word_size: int = 32) -> List[int]:
        """
        Generate a sequence of words.
        
        Args:
            length: Number of words to generate
            word_size: Size of each word in bits
            
        Returns:
            List of words (integers)
            
        Raises:
            NotImplementedError: If the generator doesn't support word generation
            ValueError: If length is not positive
        """
        if not isinstance(length, int) or length <= 0:
            raise ValueError("Length must be a positive integer.")
        
        try:
            return [self._next_word(word_size) for _ in range(length)]
        except NotImplementedError:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support word generation. "
                f"Use a generator that implements _next_word()."
            )
    
    def generate_floats(self, length: int, min_val: float = -1.0, 
                       max_val: float = 1.0) -> List[float]:
        """
        Generate a sequence of floating-point values.
        
        Args:
            length: Number of floats to generate
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)
            
        Returns:
            List of floats in [min_val, max_val]
            
        Raises:
            NotImplementedError: If the generator doesn't support float generation
            ValueError: If length is not positive
        """
        if not isinstance(length, int) or length <= 0:
            raise ValueError("Length must be a positive integer.")
        
        # Default implementation uses words to generate floats
        try:
            word_size = 32
            words = self.generate_words(length, word_size)
            max_word = (1 << word_size) - 1
            # Normalize to [0, 1] then scale to [min_val, max_val]
            return [min_val + (w / max_word) * (max_val - min_val) for w in words]
        except NotImplementedError:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support float generation. "
                f"Implement either _next_word() or override generate_floats()."
            )
    
    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        """
        Set generator parameters.
        
        Args:
            **kwargs: Generator-specific parameters
            
        Raises:
            ValueError: If parameters are invalid
        """
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current generator parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
    
    def set_seed(self, seed: int) -> None:
        """
        Set the generator seed.
        
        Args:
            seed: The seed value
        """
        self._seed = seed
        self._configure_internal()
        self.reset()
    
    def get_seed(self) -> Optional[int]:
        """Get the current seed."""
        return self._seed
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the generator to its initial state."""
    
    @abstractmethod
    def get_state(self) -> Any:
        """
        Get the current generator state.
        
        Returns:
            The current state (generator-specific format)
        """
    
    def supports_bits(self) -> bool:
        """Check if generator supports bit generation."""
        try:
            # Try to generate a single bit
            self._next_bit()
            return True
        except NotImplementedError:
            return False
        finally:
            self.reset()
    
    def supports_words(self) -> bool:
        """Check if generator supports word generation."""
        try:
            # Try to generate a single word
            self._next_word()
            return True
        except NotImplementedError:
            return False
        finally:
            self.reset()
    
    def supports_floats(self) -> bool:
        """Check if generator supports float generation."""
        try:
            # Try to generate a single float
            self.generate_floats(1)
            return True
        except NotImplementedError:
            return False
        finally:
            self.reset()


class DefaultGenerator(HDCGenerator):
    """
    Default generator using NumPy's random number generator.
    Supports all output types.
    """
    
    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize with optional seed."""
        super().__init__(seed)
    
    def _configure_internal(self) -> None:
        """Configure NumPy RNG."""
        if self._seed is not None:
            self._rng = np.random.RandomState(self._seed)
        else:
            self._rng = np.random.RandomState()
        self._state = self._rng.get_state()
    
    def _next_bit(self) -> int:
        """Generate next bit."""
        return int(self._rng.randint(0, 2))
    
    def _next_word(self, word_size: int = 32) -> int:
        """Generate next word."""
        if word_size < 64:
            # 1 << word_size fits in int64 for word_size up to 63
            return int(self._rng.randint(0, 1 << word_size, dtype=np.int64))
        else:
            # 64-bit: combine two 32-bit halves to avoid int64 overflow
            lo = int(self._rng.randint(0, 1 << 32, dtype=np.int64))
            hi = int(self._rng.randint(0, 1 << 32, dtype=np.int64))
            return (hi << 32) | lo
    
    def generate_floats(self, length: int, min_val: float = -1.0, 
                       max_val: float = 1.0) -> List[float]:
        """Generate floats efficiently using NumPy."""
        return list(self._rng.uniform(min_val, max_val, length))
    
    def set_parameters(self, **kwargs) -> None:
        """Default generator has no additional parameters."""
        if kwargs:
            warnings.warn(
                f"DefaultGenerator ignores parameters: {list(kwargs.keys())}"
            )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters."""
        return {"seed": self._seed}
    
    def reset(self) -> None:
        """Reset to initial seed state."""
        self._configure_internal()
    
    def get_state(self) -> Any:
        """Get NumPy RNG state."""
        return self._rng.get_state()
