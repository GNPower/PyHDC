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
# Threshold-based Bundling
# ============================================================================


def ElementAdditionBinaryThreshold(
    *hypervectors: ArrayLike,
    min_val: float = 0.0,
    max_val: float = 1.0,
    random_choice_range: Optional[float] = None,
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

    Returns:
        Tuple of (bundled binary hypervector, metadata dict)
        Metadata contains:
            - "random_zone_count": number of elements in random choice zone
        Note: "operation" key is added by encoding wrapper, not this function.

    Example:
        >>> v1 = np.array([1, 0, 1, 0])
        >>> v2 = np.array([1, 1, 0, 0])
        >>> v3 = np.array([1, 0, 0, 1])
        >>> result, metadata = ElementAdditionBinaryThreshold(v1, v2, v3)
        >>> # result: [1, 0, 0, 0] (1 appears in 3/3, 2/3, 1/3, 1/3 positions)
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)
    num_vectors = len(hvs)
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
        total = torch.sum(torch.stack(hvs), dim=0)

        # Track elements in random zone
        in_random_zone = torch.where(
            total > upper_bound, 0, torch.where(total < lower_bound, 0, 1)
        )
        random_zone_count = int(torch.sum(in_random_zone).item())

        # Always generate random values for the middle region
        random_vals = torch.rand(total.shape, device=total.device)
        random_choice = torch.where(random_vals < 0.5, min_val, max_val)

        # Three-way decision: above â†’ max_val, below â†’ min_val, middle â†’ random
        result = torch.where(
            total > upper_bound,
            max_val,
            torch.where(total < lower_bound, min_val, random_choice),
        )

        # Return metadata WITHOUT "operation" key - encoding wrapper adds it
        metadata = {
            "in_random_zone": in_random_zone,
            "random_zone_count": random_zone_count,
        }
        return result, metadata
    else:
        total = np.add.reduce(hvs)

        # Track elements in random zone
        in_random_zone = np.where(
            total > upper_bound, 0, np.where(total < lower_bound, 0, 1)
        )
        random_zone_count = int(np.sum(in_random_zone))

        # Always generate random values for the middle region
        random_vals = np.random.rand(*total.shape)
        random_choice = np.where(random_vals < 0.5, min_val, max_val)

        # Three-way decision: above â†’ max_val, below â†’ min_val, middle â†’ random
        result = np.where(
            total > upper_bound,
            max_val,
            np.where(total < lower_bound, min_val, random_choice),
        )

        # Return metadata WITHOUT "operation" key - encoding wrapper adds it
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
) -> Tuple[ArrayLike, Dict[str, Any]]:
    """
    Element-wise addition with bipolar thresholding.

    Bundles bipolar hypervectors {-1, +1} by thresholding at zero.
    Positive sums become max_val, negative sums become min_val.

    Coordinates whose |sum| falls within random_choice_range * sqrt(N) of zero
    (the band) are replaced by independent fair draws from {min_val, max_val}.
    Defaulting random_choice_range to 0.0 limits this to exact ties only,
    preserving the standard majority-vote behavior.

    Args:
        *hypervectors: Variable number of hypervectors to bundle, or single 2D batch
        min_val: Value for elements with negative sum
        max_val: Value for elements with non-negative sum
        random_choice_range: Optional float (rho). Coordinates with
            |sum| <= rho * sqrt(N) are randomly assigned. Defaults to 0.0 (ties only).

    Returns:
        Tuple of (bundled bipolar hypervector, metadata dict).
        Metadata contains "random_zone_count".
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
        total = np.add.reduce(hvs)
        in_band = np.abs(total) <= threshold
        random_zone_count = int(in_band.sum())
        random_vals = np.where(np.random.rand(*total.shape) < 0.5, min_val, max_val)
        result = np.where(
            total > threshold,
            max_val,
            np.where(total < -threshold, min_val, random_vals),
        )
        return result, {"random_zone_count": random_zone_count}
