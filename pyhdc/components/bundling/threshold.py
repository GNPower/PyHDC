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
# Threshold-based Bundling
# ============================================================================


def ElementAdditionBinaryThreshold(
    *hypervectors: ArrayLike,
    min_val: float = 0.0,
    max_val: float = 1.0,
    random_choice_range: Optional[float] = None,
    axis: Union[None, int, Tuple[int, ...]] = None,
) -> Tuple[ArrayLike, Dict[str, Any]]:
    """
    Element-wise addition with binary thresholding.

    Bundles binary hypervectors by majority voting. Elements that appear
    in more than half of the input vectors are set to max_val, others to min_val.

    Args:
        *hypervectors: Variable number of hypervectors to bundle, or single 2D batch
        min_val: Value for elements below threshold
        max_val: Value for elements at or above threshold
        random_choice_range: Optional float [0.0, 1.0] specifying the fraction of
            total vectors that defines the random choice zone around the threshold.
            Elements with sums outside this range are deterministically assigned,
            while elements within the range are randomly assigned.
        axis: Batch axis (or axes) to fold (defaults to the last batch axis).

    Returns:
        Tuple of (bundled binary hypervector, metadata dict). The metadata
        contains "random_zone_count" (the number of elements in the random
        choice zone). The "operation" key is added by the encoding wrapper,
        not this function.

    Example:
        >>> v1 = np.array([1, 0, 1, 0])
        >>> v2 = np.array([1, 1, 0, 0])
        >>> v3 = np.array([1, 0, 0, 1])
        >>> result, metadata = ElementAdditionBinaryThreshold(v1, v2, v3)
        >>> # result: [1, 0, 0, 0] (1 appears in 3/3, 2/3, 1/3, 1/3 positions)
    """
    batch, is_torch, _, reduce_axes = _normalize_bundling(*hypervectors, axis=axis)
    num_vectors = _reduce_count(batch, reduce_axes)
    threshold = num_vectors / 2.0

    # Default random_choice_range to 0.0 to handle ties at threshold
    if random_choice_range is None:
        random_choice_range = 0.0

    # Calculate range boundaries.
    # Use floor (int), not round: rho*sqrt(N)/2 gives the continuous band half-width
    # in {0,1} count space.  Rounding up expands the band beyond the theory threshold
    # and causes large f discrepancies at high rho (e.g. round(2.773)=3 vs int=2).
    range_size = int(random_choice_range * (np.sqrt(num_vectors) / 2))
    lower_bound = threshold - range_size
    upper_bound = threshold + range_size

    if is_torch:
        assert torch is not None  # Type narrowing for type checkers
        total = batch.sum(dim=reduce_axes)

        # Track elements in random zone
        in_random_zone = torch.where(
            total > upper_bound, 0, torch.where(total < lower_bound, 0, 1)
        )
        random_zone_count = _zone_count(in_random_zone, is_torch)

        # Always generate random values for the middle region
        random_vals = torch.rand(total.shape, device=total.device)
        random_choice = torch.where(random_vals < 0.5, min_val, max_val)

        # Three-way decision: above -> max_val, below -> min_val, middle -> random
        result = torch.where(
            total > upper_bound,
            max_val,
            torch.where(total < lower_bound, min_val, random_choice),
        )

        # Return metadata WITHOUT "operation" key, encoding wrapper adds it
        metadata = {
            "in_random_zone": in_random_zone,
            "random_zone_count": random_zone_count,
        }
        return result, metadata
    else:
        total = batch.sum(axis=reduce_axes)

        # Track elements in random zone
        in_random_zone = np.where(
            total > upper_bound, 0, np.where(total < lower_bound, 0, 1)
        )
        random_zone_count = _zone_count(in_random_zone, is_torch)

        # Always generate random values for the middle region
        random_vals = np.random.rand(*total.shape)
        random_choice = np.where(random_vals < 0.5, min_val, max_val)

        # Three-way decision: above -> max_val, below -> min_val, middle -> random
        result = np.where(
            total > upper_bound,
            max_val,
            np.where(total < lower_bound, min_val, random_choice),
        )

        # Return metadata WITHOUT "operation" key, encoding wrapper adds it
        metadata = {
            "in_random_zone": in_random_zone,
            "random_zone_count": random_zone_count,
        }
        return result, metadata


def ElementAdditionBipolarThreshold(
    *hypervectors: ArrayLike,
    min_val: float = -1.0,
    max_val: float = 1.0,
    random_choice_range: Optional[float] = None,
    axis: Union[None, int, Tuple[int, ...]] = None,
) -> Tuple[ArrayLike, Dict[str, Any]]:
    r"""
    Element-wise addition with bipolar thresholding.

    Bundles bipolar hypervectors {-1, +1} by thresholding at zero.
    Positive sums become max_val, negative sums become min_val.

    Coordinates whose \|sum\| falls within random_choice_range * sqrt(N) of zero
    (the band) are replaced by independent fair draws from {min_val, max_val}.
    Defaulting random_choice_range to 0.0 limits this to exact ties only,
    preserving the standard majority-vote behavior.

    Args:
        *hypervectors: Variable number of hypervectors to bundle, or single 2D batch
        min_val: Value for elements with negative sum
        max_val: Value for elements with non-negative sum
        random_choice_range: Optional float (rho). Coordinates with
            \|sum\| <= rho * sqrt(N) are randomly assigned. Defaults to 0.0 (ties only).
        axis: Batch axis (or axes) to fold (defaults to the last batch axis).

    Returns:
        Tuple of (bundled bipolar hypervector, metadata dict).
        Metadata contains "random_zone_count".
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
            torch.full_like(total, min_val),
            torch.full_like(total, max_val),
        )
        result = torch.where(
            total > threshold,
            torch.full_like(total, max_val),
            torch.where(
                total < -threshold, torch.full_like(total, min_val), random_vals
            ),
        )
        return result, {"random_zone_count": random_zone_count}
    else:
        total = batch.sum(axis=reduce_axes)
        in_band = np.abs(total) <= threshold
        random_zone_count = _zone_count(in_band, is_torch)
        random_vals = np.where(np.random.rand(*total.shape) < 0.5, min_val, max_val)
        result = np.where(
            total > threshold,
            max_val,
            np.where(total < -threshold, min_val, random_vals),
        )
        return result, {"random_zone_count": random_zone_count}
