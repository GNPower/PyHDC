#!/usr/bin/env python
"""
Hyperdimensional Computing (HDC) Library

A comprehensive library for hyperdimensional computing with support for
multiple backends (NumPy and PyTorch), custom generators, and various
encoding schemes.

Basic Usage:
    >>> import pyhdc
    >>>
    >>> # Create an encoding
    >>> encoding = pyhdc.MAP_C(dimension=10_000)
    >>>
    >>> # Generate hypervectors
    >>> v1 = encoding.generate()
    >>> v2 = encoding.generate()
    >>>
    >>> # Perform operations
    >>> similarity = v1.similarity(v2)
    >>> bundled = v1.bundle(v2)
    >>> bound = v1.bind(v2)

Advanced Usage:
    >>> import pyhdc
    >>> from pyhdc.generation import CommonLCGGenerators
    >>>
    >>> # Use custom generator
    >>> lcg = CommonLCGGenerators.park_miller(seed=42)
    >>> encoding = pyhdc.HRR(dimension=10_000, generator=lcg)
    >>> v = encoding.generate()
    >>>
    >>> # PyTorch backend
    >>> if pyhdc.TORCH_AVAILABLE:
    >>>     encoding_gpu = pyhdc.MAP_C(dimension=10_000, backend="torch", device="cuda")
    >>>     v_gpu = encoding_gpu.generate()
"""

# Version information
__version__ = "0.0.1"
__author__ = "GNPower"

# Submodules are available for direct import
# Recovery algorithms from recovery module
from pyhdc import components, generation, recovery

# Encodings from encodings module
from pyhdc.encodings import (
    BSC,
    BSDC_CDT,  # Binary; Holographic; MAP; Matrix
    BSDC_S,
    BSDC_SEG,
    FHRR,
    HRR,
    MAP_B,
    MAP_C,
    MAP_I,
    MBAT,
    VTB,
    HRR_ConstNorm,
    HRR_NoNorm,
    MAP_I_Bits,
)
from pyhdc.encodings.base import Encoding

# Exceptions
from pyhdc.exceptions import (
    DimensionsNotMatchingError,
    DtypesNotMatchingError,
    GeneratorNotSupportedError,
    HDCException,
    RecoveryError,
    RecoveryNotConvergedError,
    RecoveryNotSupportedError,
)

# Generators from generation module
from pyhdc.generation.base import DefaultGenerator, HDCGenerator

# Core classes and functions from hypervector module
from pyhdc.hypervector import (  # Core classes; Backend availability; Convenience functions
    TORCH_AVAILABLE,
    BackendManager,
    EncodingSpec,
    Hypervector,
    bind,
    bundle,
    generate,
    zeros,
)

# Type aliases for convenience
from pyhdc.types import ArrayLike, Backend, Device

# Export all public items
__all__ = [
    # Version
    "__version__",
    "__author__",
    # Core classes
    "Hypervector",
    "Encoding",
    "HDCGenerator",
    "DefaultGenerator",
    "BackendManager",
    "EncodingSpec",
    # Backend support
    "TORCH_AVAILABLE",
    # Encoding implementations
    "MAP_C",
    "MAP_I",
    "MAP_I_Bits",
    "MAP_B",
    "HRR",
    "HRR_NoNorm",
    "HRR_ConstNorm",
    "VTB",
    "MBAT",
    "BSC",
    "BSDC_CDT",
    "BSDC_S",
    "BSDC_SEG",
    "FHRR",
    # Convenience functions
    "generate",
    "zeros",
    "bundle",
    "bind",
    # Exceptions
    "HDCException",
    "DimensionsNotMatchingError",
    "DtypesNotMatchingError",
    "GeneratorNotSupportedError",
    "RecoveryError",
    "RecoveryNotConvergedError",
    "RecoveryNotSupportedError",
    # Type aliases
    "ArrayLike",
    "Backend",
    "Device",
    # Submodules
    "components",
    "generation",
    "recovery",
]
