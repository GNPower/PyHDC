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
__version__ = "2.1.0"
__author__ = "GNPower"

# Submodules are available for direct import
from pyhdc import components

# Global backend/device preferences
from pyhdc.config import (
    get_default_backend,
    get_default_device,
    prefer_cpu,
    prefer_cuda,
    prefer_numpy,
    prefer_torch,
)

# Encodings from encodings module
from pyhdc.encodings import (  # Binary; Holographic; MAP; Matrix
    BSC,
    BSDC_CDT,
    BSDC_S,
    BSDC_SEG,
    BSDC_THIN,
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
from pyhdc.hypervector import (
    TORCH_AVAILABLE,
    BackendManager,
    EncodingSpec,
    Hypervector,
    bind,
    bundle,
    generate,
    inverse,
    negative,
    normalize,
    permute,
    stack,
    unbind,
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
    "BSDC_THIN",
    "FHRR",
    # Convenience functions
    "generate",
    "zeros",
    "bundle",
    "bind",
    "unbind",
    "stack",
    "permute",
    "inverse",
    "negative",
    "normalize",
    # Global backend/device preferences
    "prefer_torch",
    "prefer_cuda",
    "prefer_numpy",
    "prefer_cpu",
    "get_default_backend",
    "get_default_device",
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
