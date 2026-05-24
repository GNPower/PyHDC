# PyHDC

[![PyPI version](https://img.shields.io/pypi/v/PyHDC.svg)](https://pypi.org/project/PyHDC/)
[![Tests](https://github.com/GNPower/PyHDC/actions/workflows/test.yml/badge.svg)](https://github.com/GNPower/PyHDC/actions/workflows/test.yml)
[![Coverage](https://codecov.io/gh/GNPower/PyHDC/branch/main/graph/badge.svg)](https://codecov.io/gh/GNPower/PyHDC)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/pypi/pyversions/PyHDC.svg)](https://pypi.org/project/PyHDC/)

**A Python library for Hyperdimensional Computing (HDC) and Vector Symbolic Architectures (VSA).**

Full documentation: [https://pyhdc.readthedocs.io/en/latest/](https://pyhdc.readthedocs.io/en/latest/)

---

## Overview

PyHDC provides a unified interface for working with high-dimensional binary and continuous vectors used in HDC/VSA-based computing. It supports multiple encoding schemes, pluggable pseudorandom generators, and both NumPy and PyTorch backends.

## Installation

```bash
pip install PyHDC
```

## Quick Start

```python
import pyhdc as hdc

# Create an encoding
enc = hdc.MAP_C(dimension=10_000)

# Generate hypervectors
v1 = enc.generate()
v2 = enc.generate()

# Core operations
bundled = v1.bundle(v2)        # superposition
bound   = v1.bind(v2)          # association
sim     = v1.similarity(v2)    # similarity score
```

## Features

- **14 encoding schemes**: MAP-C, MAP-I, MAP-I Bits, MAP-B, HRR, HRR (no norm), HRR (const norm), FHRR, VTB, MBAT, BSC, BSDC-CDT, BSDC-S, BSDC-SEG
- **7 generator families**: LCG, Fibonacci/Galois LFSR, DLFSR, LCA, PCG, Xorshift, Shifted Counter
- **NumPy and PyTorch backends** with optional GPU support
- **Composable operations**: bind, bundle, unbind, similarity, thinning
- **Custom encodings**: extend `Encoding` with your own `EncodingSpec`
- **Type-annotated** throughout

## License

MIT [LICENSE](LICENSE).
