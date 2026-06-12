from math import sqrt
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Optional PyTorch support
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _normalize_bundling

# Type aliases
from pyhdc.types import ArrayLike

# ============================================================================
# Element Addition with Normalization Operations
# ============================================================================


def ElementAdditionNormalized(
    *hypervectors: ArrayLike, random_choice_range: Optional[float] = None
) -> Tuple[ArrayLike, Dict[str, Any]]:
    """
    Element-wise addition with normalization to unit length.

    Bundles hypervectors by summing elements and normalizing the result
    to have magnitude 1. Used in HRR and related encodings.

    HRR elements are N(0, 1/D) per Schlegel et al. (2022), so the
    pre-aggregate T_k = sum of N iid N(0, 1/D) values has std sqrt(N/D).
    When random_choice_range (rho) is set, coordinates whose |T_k| <=
    rho * sqrt(N/D) are replaced by independent N(0, 1/D) draws before
    normalization (band randomization). This keeps fresh draws at the same
    scale as individual input elements, consistent with the GU/GB theory.

    Args:
        *hypervectors: Variable number of hypervectors to bundle, or single 2D batch
        random_choice_range: Optional float (rho). Coordinates with
            |sum| <= rho * sqrt(N/D) are randomly replaced. Defaults to 0.0.

    Returns:
        Tuple of (bundled and normalized hypervector, metadata dict).
        Metadata contains "random_zone_count".

    Note:
        Result will have ||result|| = 1
    """
    batch, is_torch, _ = _normalize_bundling(*hypervectors)
    num_vectors = batch.shape[1]

    if random_choice_range is None:
        random_choice_range = 0.0

    if is_torch:
        assert torch is not None
        total = batch.sum(dim=1)
        D = total.shape[0]
        # sigma_N for N(0,1/D) elements: Var[T_k] = N/D, so sigma_N = sqrt(N/D)
        threshold = random_choice_range * sqrt(num_vectors / D)
        in_band = torch.abs(total) <= threshold
        random_zone_count = int(in_band.sum().item())
        # Fresh draws at element scale N(0, 1/D)
        random_vals = torch.randn(total.shape, device=total.device) / sqrt(D)
        total_modified = torch.where(in_band, random_vals, total)
        norm = torch.norm(total_modified)
        return total_modified / norm, {"random_zone_count": random_zone_count}
    else:
        total = batch.sum(axis=1)
        D = len(total)
        # sigma_N for N(0,1/D) elements: Var[T_k] = N/D, so sigma_N = sqrt(N/D)
        threshold = random_choice_range * sqrt(num_vectors / D)
        in_band = np.abs(total) <= threshold
        random_zone_count = int(in_band.sum())
        # Fresh draws at element scale N(0, 1/D)
        random_vals = (np.random.randn(*total.shape) / sqrt(D)).astype(total.dtype)
        total_modified = np.where(in_band, random_vals, total)
        norm = np.linalg.norm(total_modified)
        return total_modified / norm, {"random_zone_count": random_zone_count}


def ElementAdditionConstantNormalized(*hypervectors: ArrayLike) -> ArrayLike:
    """
    Element-wise addition with approximate constant normalization.

    Bundles hypervectors by summing elements and dividing by sqrt(M) where
    M is the number of vectors bundled. Provides approximate unit length
    without computing the full norm (faster).

    Args:
        *hypervectors: Variable number of hypervectors to bundle, or single 2D batch

    Returns:
        Bundled and approximately normalized hypervector

    Note:
        Result will have ||result|| â‰ˆ 1 (approximate)
    """
    batch, is_torch, _ = _normalize_bundling(*hypervectors)
    M = batch.shape[1]
    norm_factor = sqrt(M)

    if is_torch:
        total = batch.sum(dim=1)
        return total / norm_factor
    else:
        total = batch.sum(axis=1)
        return total / norm_factor
