from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Optional PyTorch support
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import (
    _normalize_bundling,
    _reduce_count,
    _zone_count,
)

# Type aliases
from pyhdc.types import ArrayLike

# ============================================================================
# Angle Operations
# ============================================================================


def AnglesOfElementAddition(
    *hypervectors: ArrayLike,
    random_choice_range: Optional[float] = None,
    axis: Union[None, int, Tuple[int, ...]] = None,
) -> Tuple[ArrayLike, Dict[str, Any]]:
    r"""
    Bundling by adding phase angles (for FHRR).

    Bundles phase-encoded hypervectors by computing the mean angle.
    Converts angles to complex numbers, sums them, and extracts the
    resulting phase. Used with Fourier Holographic Reduced Representations.

    When random_choice_range is set, coordinates whose phasor magnitude
    \|sum\| falls within rho * sqrt(N/2) are replaced by independent fair
    draws from Uniform[-pi, pi] (band randomization for FHRR). The
    Rayleigh sigma of the neutral phasor sum magnitude is sqrt(N/2).
    Defaulting random_choice_range to 0.0 limits this to near-zero
    magnitude sums only.

    Args:
        *hypervectors: Variable number of phase hypervectors, or single 2D batch
        random_choice_range: Optional float (rho). Coordinates with phasor
            magnitude <= rho * sqrt(N/2) are randomly assigned. Defaults to 0.0.
        axis: Batch axis (or axes) to fold (defaults to the last batch axis).

    Returns:
        Tuple of (bundled phase hypervector, metadata dict).
        Metadata contains "random_zone_count".

    Note:
        Input values should be angles in radians. Output is also in radians.
    """
    batch, is_torch, _, reduce_axes = _normalize_bundling(*hypervectors, axis=axis)
    num_vectors = _reduce_count(batch, reduce_axes)

    if random_choice_range is None:
        random_choice_range = 0.0

    # sigma_N for FHRR: sum of N unit phasors has Rayleigh sigma = sqrt(N/2)
    threshold = random_choice_range * np.sqrt(num_vectors / 2.0)

    if is_torch:
        assert torch is not None
        real = torch.cos(batch).sum(dim=reduce_axes)
        imag = torch.sin(batch).sum(dim=reduce_axes)
        magnitude = torch.sqrt(real**2 + imag**2)
        in_band = magnitude <= threshold
        random_zone_count = _zone_count(in_band, is_torch)
        random_angles = torch.rand(real.shape, device=real.device) * (2 * np.pi) - np.pi
        result = torch.where(in_band, random_angles, torch.atan2(imag, real))
        return result, {"random_zone_count": random_zone_count}
    else:
        real = np.cos(batch).sum(axis=reduce_axes)
        imag = np.sin(batch).sum(axis=reduce_axes)
        magnitude = np.sqrt(real**2 + imag**2)
        in_band = magnitude <= threshold
        random_zone_count = _zone_count(in_band, is_torch)
        random_angles = np.random.uniform(-np.pi, np.pi, real.shape).astype(real.dtype)
        result = np.where(in_band, random_angles, np.arctan2(imag, real))
        return result, {"random_zone_count": random_zone_count}
