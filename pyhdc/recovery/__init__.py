#!/usr/bin/env python
"""
HDC Recovery Module

Algorithms for recovering generator seeds and structures from observed output
sequences. Full implementations are deferred to a future release.
"""

_NOT_AVAILABLE = (
    "The recovery module is not yet available in this release. "
    "It will be included in a future version of PyHDC."
)


def __getattr__(name: str) -> object:
    raise NotImplementedError(_NOT_AVAILABLE)