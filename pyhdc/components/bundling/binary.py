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
# Binary Operations
# ============================================================================


def Disjunction(*hypervectors: ArrayLike) -> ArrayLike:
    """
    Bitwise OR bundling for sparse binary vectors.

    Bundles sparse binary hypervectors using bitwise OR. An element is 1
    in the result if it is 1 in any input vector. Preserves sparsity better
    than addition for Binary Sparse Distributed Codes.

    Args:
        *hypervectors: Variable number of sparse binary hypervectors, or single 2D batch

    Returns:
        Bundled sparse binary hypervector

    Example:
        >>> v1 = np.array([1, 0, 1, 0])
        >>> v2 = np.array([0, 1, 0, 0])
        >>> result = Disjunction(v1, v2)
        >>> # result: [1, 1, 1, 0]
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)

    if is_torch:
        return torch.stack([hv.bool() for hv in hvs], dim=0).any(dim=0).to(hvs[0].dtype)
    else:
        return np.bitwise_or.reduce(hvs).astype(hvs[0].dtype)


def DisjunctionThinned(*hypervectors: ArrayLike, density: float = 0.5) -> ArrayLike:
    """
    Bitwise OR bundling with random thinning to maintain density.

    Bundles sparse binary hypervectors using bitwise OR, then randomly zeros
    bits to keep the fraction of 1-bits at most `density`.

    Args:
        *hypervectors: Variable number of sparse binary hypervectors, or single 2D batch
        density: Maximum output density (fraction of 1-bits), defaults to 0.5

    Returns:
        Bundled and thinned sparse binary hypervector

    Example:
        >>> v1 = np.array([1, 0, 1, 0])
        >>> v2 = np.array([0, 1, 1, 0])
        >>> result = DisjunctionThinned(v1, v2, density=0.25)
        >>> # result has at most 1 nonzero element (25% of 4)
    """
    from math import ceil

    hvs, is_torch, _ = _normalize_inputs(*hypervectors)

    if is_torch:
        bundled = (
            torch.stack([hv.bool() for hv in hvs], dim=0).any(dim=0).to(hvs[0].dtype)
        )

        num_nonzero = ceil(bundled.numel() * density)
        indices = torch.nonzero(bundled, as_tuple=True)[0]
        if num_nonzero >= indices.numel():
            return bundled
        perm = torch.randperm(indices.numel())[:num_nonzero]
        kept = indices[perm]
        result = torch.zeros_like(bundled)
        result[kept] = 1
        return result
    else:
        bundled = np.bitwise_or.reduce(hvs).astype(hvs[0].dtype)

        num_nonzero = ceil(bundled.size * density)
        indices = np.nonzero(bundled)[0]
        if num_nonzero >= indices.size:
            return bundled
        kept = np.random.choice(indices, size=num_nonzero, replace=False)
        result = np.zeros_like(bundled)
        result[kept] = 1
        return result
