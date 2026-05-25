#!/usr/bin/env python
"""
Common Encodings for Hyperdimensional Computing

This module provides various common encodings for hyperdimensional
computing. These encodings can be used to create hypervectors
with specific properties.

Available encodings:
- MAP-C: Multiplicative Additive Permutation for Continuous Hypervectors
- MAP-I: Multiplicative Additive Permutation for Integer Hypervectors
- MAP-I-Bits: MAP-I variant using custom integer bitwidths
- HRR: Holographic Reduced Representations for Continuous Hypervectors
- HRR-NoNorm: HRR variant without normalization
- HRR-ConstNorm: HRR variant with constant normalization by sqrt(M), where M is the number of vectors bundled
- VTB: Vector-derived Transformation Binding for Continuous Hypervectors
- MBAT: Matrix Binding of Additive Terms for Continuous Hypervectors
- MAP-B: Multiplicative Additive Permutation for Binary Hypervectors
- BSC: Binary Sparse Coding for Sparse Binary Hypervectors
- BSDC-CDT: Binary Sparse Distributed Coding with Count Distribution for Sparse Binary Hypervectors
- BSDC-S: Binary Sparse Distributed Coding with Sparse Distribution for Sparse Binary Hypervectors
- BSDC-SEG: Binary Sparse Distributed Coding with Sparse Segmented Distribution for Sparse Binary Hypervectors
- BSDC-THIN: BSDC with post-bundling random thinning to maintain density constraint
- FHRR: Fourier Holographic Reduced Representations for Complex Hypervectors
"""

# Optional PyTorch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


from pyhdc.encodings.binary import BSC, BSDC_CDT, BSDC_S, BSDC_SEG, BSDC_THIN
from pyhdc.encodings.holographic import FHRR, HRR, HRR_ConstNorm, HRR_NoNorm
from pyhdc.encodings.map import MAP_B, MAP_C, MAP_I, MAP_I_Bits
from pyhdc.encodings.matrix import MBAT, VTB

# Export all encodings
__all__ = [
    # Binary
    "BSC",
    "BSDC_CDT",
    "BSDC_S",
    "BSDC_SEG",
    "BSDC_THIN",
    # Holographic
    "HRR",
    "HRR_NoNorm",
    "HRR_ConstNorm",
    "FHRR",
    # MAP
    "MAP_C",
    "MAP_I",
    "MAP_I_Bits",
    "MAP_B",
    # Matrix
    "VTB",
    "MBAT",
]
