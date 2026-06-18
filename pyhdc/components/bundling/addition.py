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
# Element Addition Operations
# ============================================================================


def ElementAddition(
    *hypervectors: ArrayLike,
    random_choice_range: Optional[float] = None,
    axis: Union[None, int, Tuple[int, ...]] = None,
) -> Tuple[ArrayLike, Dict[str, Any]]:
    r"""
    Element-wise addition bundling.

    Bundles hypervectors by summing corresponding elements. The simplest
    bundling operation that preserves information from all inputs.

    When random_choice_range is set, coordinates whose \|sum\| falls within
    rho * sqrt(N) are replaced by independent fair draws from {-1, +1}
    (band randomization for MAP_I bipolar integer encoding). Defaulting
    random_choice_range to 0.0 limits this to exact zero-sums only.

    Args:
        *hypervectors: Variable number of hypervectors to bundle, or single 2D batch
        random_choice_range: Optional float (rho). Coordinates with
            \|sum\| <= rho * sqrt(N) are randomly assigned. Defaults to 0.0
            (zero-ties only).
        axis: Batch axis (or axes) to fold. Defaults to the last batch axis, so a
            ``(D, N)`` batch collapses to ``(D,)`` and a ``(D, N, M)`` batch
            to ``(D, N)``.

    Returns:
        Tuple of (bundled hypervector, metadata dict).
        Metadata contains "random_zone_count".

    Example:
        >>> v1 = np.array([1, -1, 1, -1])
        >>> v2 = np.array([1, 1, -1, -1])
        >>> result, _ = ElementAddition(v1, v2)
        >>> # result: [2, 0, 0, -2]
    """
    batch, is_torch, _, reduce_axes = _normalize_bundling(*hypervectors, axis=axis)
    num_vectors = _reduce_count(batch, reduce_axes)

    if random_choice_range is None:
        random_choice_range = 0.0

    threshold = random_choice_range * np.sqrt(num_vectors)

    if is_torch:
        assert torch is not None
        total = batch.sum(dim=reduce_axes)
        in_band = torch.abs(total) <= threshold
        random_zone_count = _zone_count(in_band, is_torch)
        random_vals = torch.where(
            torch.rand(total.shape, device=total.device) < 0.5,
            torch.full_like(total, -1.0),
            torch.full_like(total, 1.0),
        )
        result = torch.where(in_band, random_vals, total)
        return result, {"random_zone_count": random_zone_count}
    else:
        total = batch.sum(axis=reduce_axes)
        in_band = np.abs(total) <= threshold
        random_zone_count = _zone_count(in_band, is_torch)
        random_vals = np.where(np.random.rand(*total.shape) < 0.5, -1, 1).astype(
            total.dtype
        )
        result = np.where(in_band, random_vals, total)
        return result, {"random_zone_count": random_zone_count}


def ElementAdditionBits(
    *hypervectors: ArrayLike,
    min_val: int = -1,
    max_val: int = 1,
    axis: Union[None, int, Tuple[int, ...]] = None,
) -> ArrayLike:
    """
    Element-wise addition with per-step clipping (bit-limited).

    Bundles hypervectors by iteratively adding and clipping after each addition.
    Useful for fixed-point or integer arithmetic where overflow must be prevented.

    Args:
        *hypervectors: Variable number of hypervectors to bundle, or single 2D batch
        min_val: Minimum element value (clipped after each addition)
        max_val: Maximum element value (clipped after each addition)
        axis: Single batch axis to fold (defaults to the last batch axis). The
            per-step clip is order-dependent, so a tuple of axes is not supported.

    Returns:
        Bundled hypervector with per-step clipping
    """
    batch, is_torch, _, reduce_axes = _normalize_bundling(*hypervectors, axis=axis)
    if len(reduce_axes) != 1:
        raise ValueError("ElementAdditionBits supports reducing a single axis only")
    reduce_axis = reduce_axes[0]
    num_vectors = batch.shape[reduce_axis]

    # The per-step clip only changes the result when a partial sum leaves
    # [min_val, max_val]. Partial sums are bounded by +/- the L1 norm along
    # the reduce axis, so when that bound is within the limits the saturating
    # accumulation equals a plain sum, computed vectorized instead of looping.
    if is_torch:
        total = batch.sum(dim=reduce_axis, dtype=batch.dtype)
        l1 = batch.abs().sum(dim=reduce_axis)
        if bool((l1 <= max_val).all()) and bool(((-l1) >= min_val).all()):
            return total
        result = torch.zeros_like(torch.select(batch, reduce_axis, 0))
        for j in range(num_vectors):
            result = result + torch.select(batch, reduce_axis, j)
            result = torch.clamp(result, min_val, max_val)
        return result

    total = batch.sum(axis=reduce_axis, dtype=batch.dtype)
    l1 = np.abs(batch).sum(axis=reduce_axis)
    if (l1 <= max_val).all() and ((-l1) >= min_val).all():
        return total
    result = np.zeros_like(np.take(batch, 0, axis=reduce_axis))
    for j in range(num_vectors):
        result = np.add(result, np.take(batch, j, axis=reduce_axis))
        result = np.clip(result, min_val, max_val)
    return result


def ElementAdditionCut(
    *hypervectors: ArrayLike,
    min_val: float = -1.0,
    max_val: float = 1.0,
    random_choice_range: Optional[float] = None,
    axis: Union[None, int, Tuple[int, ...]] = None,
) -> Tuple[ArrayLike, Dict[str, Any]]:
    r"""
    Element-wise addition with clipping.

    Bundles hypervectors by summing elements and clipping the result to
    [min_val, max_val] range. Prevents unbounded growth in element values.

    When random_choice_range is set, coordinates whose \|sum\| falls within
    rho * sqrt(N) * element_std (where element_std = 1/sqrt(3) for Uniform[-1,1])
    are replaced by independent fair draws from Uniform[min_val, max_val]
    (band randomization for MAP_C continuous encoding). Defaulting
    random_choice_range to 0.0 limits this to exact zero-sums only.

    Args:
        *hypervectors: Variable number of hypervectors to bundle, or single 2D batch
        min_val: Minimum element value after bundling
        max_val: Maximum element value after bundling
        random_choice_range: Optional float (rho). Coordinates with
            \|sum\| <= rho * sqrt(N/3) are randomly assigned. Defaults to 0.0
            (zero-ties only).
        axis: Batch axis (or axes) to fold (defaults to the last batch axis).

    Returns:
        Tuple of (bundled and clipped hypervector, metadata dict).
        Metadata contains "random_zone_count".
    """
    batch, is_torch, _, reduce_axes = _normalize_bundling(*hypervectors, axis=axis)
    num_vectors = _reduce_count(batch, reduce_axes)

    if random_choice_range is None:
        random_choice_range = 0.0

    # sigma_N for Uniform[-1,1] elements: Var[X] = 1/3, so sigma_N = sqrt(N/3)
    threshold = random_choice_range * np.sqrt(num_vectors / 3.0)

    if is_torch:
        assert torch is not None
        total = batch.sum(dim=reduce_axes)
        in_band = torch.abs(total) <= threshold
        random_zone_count = _zone_count(in_band, is_torch)
        random_vals = (
            torch.rand(total.shape, device=total.device) * (max_val - min_val) + min_val
        )
        result = torch.where(in_band, random_vals, torch.clamp(total, min_val, max_val))
        return result, {"random_zone_count": random_zone_count}
    else:
        total = batch.sum(axis=reduce_axes)
        in_band = np.abs(total) <= threshold
        random_zone_count = _zone_count(in_band, is_torch)
        random_vals = np.random.uniform(min_val, max_val, total.shape).astype(
            total.dtype
        )
        result = np.where(in_band, random_vals, np.clip(total, min_val, max_val))
        return result, {"random_zone_count": random_zone_count}
