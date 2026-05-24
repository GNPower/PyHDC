from typing import Any, Dict, Optional, Tuple

import numpy as np

# Optional PyTorch support
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _normalize_inputs

# Type aliases
from pyhdc.types import ArrayLike

# ============================================================================
# Element Addition Operations
# ============================================================================


def ElementAddition(
    *hypervectors: ArrayLike, random_choice_range: Optional[float] = None
) -> Tuple[ArrayLike, Dict[str, Any]]:
    """
    Element-wise addition bundling.

    Bundles hypervectors by summing corresponding elements. The simplest
    bundling operation that preserves information from all inputs.

    When random_choice_range is set, coordinates whose |sum| falls within
    rho * sqrt(N) are replaced by independent fair draws from {-1, +1}
    (band randomization for MAP_I bipolar integer encoding). Defaulting
    random_choice_range to 0.0 limits this to exact zero-sums only.

    Args:
        *hypervectors: Variable number of hypervectors to bundle, or single 2D batch
        random_choice_range: Optional float (rho). Coordinates with
            |sum| <= rho * sqrt(N) are randomly assigned. Defaults to 0.0 (zero-ties only).

    Returns:
        Tuple of (bundled hypervector, metadata dict).
        Metadata contains "random_zone_count".

    Example:
        >>> v1 = np.array([1, -1, 1, -1])
        >>> v2 = np.array([1, 1, -1, -1])
        >>> result, _ = ElementAddition(v1, v2)
        >>> # result: [2, 0, 0, -2]
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)
    num_vectors = len(hvs)

    if random_choice_range is None:
        random_choice_range = 0.0

    threshold = random_choice_range * np.sqrt(num_vectors)

    if is_torch:
        assert torch is not None
        total = torch.sum(torch.stack(hvs), dim=0)
        in_band = torch.abs(total) <= threshold
        random_zone_count = int(in_band.sum().item())
        random_vals = torch.where(
            torch.rand(total.shape, device=total.device) < 0.5,
            torch.full_like(total, -1.0),
            torch.full_like(total, 1.0),
        )
        result = torch.where(in_band, random_vals, total)
        return result, {"random_zone_count": random_zone_count}
    else:
        total = np.add.reduce(hvs)
        in_band = np.abs(total) <= threshold
        random_zone_count = int(in_band.sum())
        random_vals = np.where(np.random.rand(*total.shape) < 0.5, -1, 1).astype(
            total.dtype
        )
        result = np.where(in_band, random_vals, total)
        return result, {"random_zone_count": random_zone_count}


def ElementAdditionBits(
    *hypervectors: ArrayLike, min_val: int = -1, max_val: int = 1
) -> ArrayLike:
    """
    Element-wise addition with per-step clipping (bit-limited).

    Bundles hypervectors by iteratively adding and clipping after each addition.
    Useful for fixed-point or integer arithmetic where overflow must be prevented.

    Args:
        *hypervectors: Variable number of hypervectors to bundle, or single 2D batch
        min_val: Minimum element value (clipped after each addition)
        max_val: Maximum element value (clipped after each addition)

    Returns:
        Bundled hypervector with per-step clipping
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)

    if is_torch:
        result = torch.zeros_like(hvs[0])
        for hv in hvs:
            result = result + hv
            result = torch.clamp(result, min_val, max_val)
    else:
        result = np.zeros_like(hvs[0])
        for hv in hvs:
            result = np.add(result, hv)
            result = np.clip(result, min_val, max_val)

    return result


def ElementAdditionCut(
    *hypervectors: ArrayLike,
    min_val: float = -1.0,
    max_val: float = 1.0,
    random_choice_range: Optional[float] = None,
) -> Tuple[ArrayLike, Dict[str, Any]]:
    """
    Element-wise addition with clipping.

    Bundles hypervectors by summing elements and clipping the result to
    [min_val, max_val] range. Prevents unbounded growth in element values.

    When random_choice_range is set, coordinates whose |sum| falls within
    rho * sqrt(N) * element_std (where element_std = 1/sqrt(3) for Uniform[-1,1])
    are replaced by independent fair draws from Uniform[min_val, max_val]
    (band randomization for MAP_C continuous encoding). Defaulting
    random_choice_range to 0.0 limits this to exact zero-sums only.

    Args:
        *hypervectors: Variable number of hypervectors to bundle, or single 2D batch
        min_val: Minimum element value after bundling
        max_val: Maximum element value after bundling
        random_choice_range: Optional float (rho). Coordinates with
            |sum| <= rho * sqrt(N/3) are randomly assigned. Defaults to 0.0 (zero-ties only).

    Returns:
        Tuple of (bundled and clipped hypervector, metadata dict).
        Metadata contains "random_zone_count".
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)
    num_vectors = len(hvs)

    if random_choice_range is None:
        random_choice_range = 0.0

    # sigma_N for Uniform[-1,1] elements: Var[X] = 1/3, so sigma_N = sqrt(N/3)
    threshold = random_choice_range * np.sqrt(num_vectors / 3.0)

    if is_torch:
        assert torch is not None
        total = torch.sum(torch.stack(hvs), dim=0)
        in_band = torch.abs(total) <= threshold
        random_zone_count = int(in_band.sum().item())
        random_vals = (
            torch.rand(total.shape, device=total.device) * (max_val - min_val) + min_val
        )
        result = torch.where(in_band, random_vals, torch.clamp(total, min_val, max_val))
        return result, {"random_zone_count": random_zone_count}
    else:
        total = np.add.reduce(hvs)
        in_band = np.abs(total) <= threshold
        random_zone_count = int(in_band.sum())
        random_vals = np.random.uniform(min_val, max_val, total.shape).astype(
            total.dtype
        )
        result = np.where(in_band, random_vals, np.clip(total, min_val, max_val))
        return result, {"random_zone_count": random_zone_count}
