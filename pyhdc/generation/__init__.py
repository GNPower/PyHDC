#!/usr/bin/env python
"""
Random Number Generators for HDC

This module provides various random number generators optimized for
hyperdimensional computing. These generators can be used to create
hypervectors with specific statistical properties.

Available generators:
- LCG: Linear Congruential Generators
- DLFSR: Digit-Serial Linear Feedback Shift Registers
- LFSR: Traditional Linear Feedback Shift Registers
- LCA: Linear Cellular Automata
- PCG: Permuted Congruential Generators
- Xorshift: Xorshift family of generators
- Shifted Counter: Counter-based generators with cryptographic mappings
"""

from pyhdc.generation.dlfsr import (
    CommonDLFSRGenerators,
    DLFSRGenerator,
    FibonacciDLFSRGenerator,
    GaloisDLFSRGenerator,
    MatrixDLFSRGenerator,
)
from pyhdc.generation.lca import (
    CommonLCAGenerators,
    ElementaryLCAGenerator,
    LCAGenerator,
    TotalisticLCAGenerator,
)
from pyhdc.generation.lcg import (
    CommonLCGGenerators,
    LCGGenerator,
    MultiplicativeLCGGenerator,
)
from pyhdc.generation.lfsr import (
    CommonLFSRGenerators,
    FibonacciLFSRGenerator,
    GaloisLFSRGenerator,
    LFSRGenerator,
)
from pyhdc.generation.pcg import (
    CommonPCGGenerators,
    MultiplicativePCGGenerator,
    PCGGenerator,
)
from pyhdc.generation.shifted_counter import (
    ARXCounterGenerator,
    CommonCounterGenerators,
    CustomMappingCounterGenerator,
    FeistelCounterGenerator,
    ShiftedCounterGenerator,
    SPNCounterGenerator,
)
from pyhdc.generation.xorshift import (
    CommonXorshiftGenerators,
    SplitMix64Generator,
    Xoroshiro128PlusGenerator,
    Xoroshiro128StarStarGenerator,
    Xorshift32Generator,
    Xorshift64Generator,
    Xorshift128Generator,
    XorshiftGenerator,
    XorshiftPlusGenerator,
    XorshiftStarGenerator,
    Xoshiro256StarStarGenerator,
    splitmix64_seed,
)

# Export all generators
__all__ = [
    # Base HDC Generator is in hypervector module
    # LCG generators
    "LCGGenerator",
    "MultiplicativeLCGGenerator",
    "CommonLCGGenerators",
    # DLFSR generators
    "DLFSRGenerator",
    "FibonacciDLFSRGenerator",
    "GaloisDLFSRGenerator",
    "MatrixDLFSRGenerator",
    "CommonDLFSRGenerators",
    # LFSR generators
    "LFSRGenerator",
    "FibonacciLFSRGenerator",
    "GaloisLFSRGenerator",
    "CommonLFSRGenerators",
    # LCA generators
    "LCAGenerator",
    "ElementaryLCAGenerator",
    "TotalisticLCAGenerator",
    "CommonLCAGenerators",
    # PCG generators
    "PCGGenerator",
    "MultiplicativePCGGenerator",
    "CommonPCGGenerators",
    # Xorshift generators
    "XorshiftGenerator",
    "Xorshift32Generator",
    "Xorshift64Generator",
    "Xorshift128Generator",
    "XorshiftPlusGenerator",
    "XorshiftStarGenerator",
    "Xoshiro256StarStarGenerator",
    "Xoroshiro128PlusGenerator",
    "Xoroshiro128StarStarGenerator",
    "SplitMix64Generator",
    "CommonXorshiftGenerators",
    "splitmix64_seed",
    # Counter-based generators
    "ShiftedCounterGenerator",
    "FeistelCounterGenerator",
    "ARXCounterGenerator",
    "SPNCounterGenerator",
    "CustomMappingCounterGenerator",
    "CommonCounterGenerators",
]
