from math import ceil
from typing import Tuple, Union

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
# Binary Operations
# ============================================================================


def Disjunction(
    *hypervectors: ArrayLike, axis: Union[None, int, Tuple[int, ...]] = None
) -> ArrayLike:
    """
    Bitwise OR bundling for sparse binary vectors.

    Bundles sparse binary hypervectors using bitwise OR. An element is 1
    in the result if it is 1 in any input vector. Preserves sparsity better
    than addition for Binary Sparse Distributed Codes.

    Args:
        *hypervectors: Variable number of sparse binary hypervectors, or single 2D batch
        axis: Batch axis (or axes) to fold (defaults to the last batch axis).

    Returns:
        Bundled sparse binary hypervector

    Example:
        >>> v1 = np.array([1, 0, 1, 0])
        >>> v2 = np.array([0, 1, 0, 0])
        >>> result = Disjunction(v1, v2)
        >>> # result: [1, 1, 1, 0]
    """
    batch, is_torch, _, reduce_axes = _normalize_bundling(*hypervectors, axis=axis)

    if is_torch:
        return torch.amax((batch != 0).to(batch.dtype), dim=reduce_axes)
    else:
        return np.bitwise_or.reduce(batch, axis=reduce_axes).astype(batch.dtype)


def DisjunctionThinned(
    *hypervectors: ArrayLike,
    density: float = 0.5,
    axis: Union[None, int, Tuple[int, ...]] = None,
) -> ArrayLike:
    """
    Bitwise OR bundling with random thinning to maintain density.

    Bundles sparse binary hypervectors using bitwise OR, then randomly zeros
    bits to keep the fraction of 1-bits at most `density`. For a batched result
    each output hypervector (column over the surviving batch axes) is thinned
    independently to ``ceil(D * density)`` set bits.

    Args:
        *hypervectors: Variable number of sparse binary hypervectors, or single 2D batch
        density: Maximum output density (fraction of 1-bits), defaults to 0.5
        axis: Single batch axis to fold (defaults to the last batch axis). Thinning
            is per-column, so a tuple of axes is not supported.

    Returns:
        Bundled and thinned sparse binary hypervector

    Example:
        >>> v1 = np.array([1, 0, 1, 0])
        >>> v2 = np.array([0, 1, 1, 0])
        >>> result = DisjunctionThinned(v1, v2, density=0.25)
        >>> # result has at most 1 nonzero element (25% of 4)
    """
    batch, is_torch, _, reduce_axes = _normalize_bundling(*hypervectors, axis=axis)
    if len(reduce_axes) != 1:
        raise ValueError("DisjunctionThinned supports reducing a single axis only")
    reduce_axis = reduce_axes[0]

    if is_torch:
        bundled = torch.amax((batch != 0).to(batch.dtype), dim=reduce_axis)
    else:
        bundled = np.bitwise_or.reduce(batch, axis=reduce_axis).astype(batch.dtype)

    # Rank-1 result: the exact 2.0 flat path (preserves RNG consumption).
    if getattr(bundled, "ndim", 1) == 1:
        if is_torch:
            num_nonzero = ceil(bundled.numel() * density)
            indices = torch.nonzero(bundled, as_tuple=True)[0]
            if num_nonzero >= indices.numel():
                return bundled
            perm = torch.randperm(indices.numel())[:num_nonzero]
            kept = indices[perm]
            result = torch.zeros_like(bundled)
            result[kept] = 1
            return result
        num_nonzero = ceil(bundled.size * density)
        indices = np.nonzero(bundled)[0]
        if num_nonzero >= indices.size:
            return bundled
        kept = np.random.choice(indices, size=num_nonzero, replace=False)
        result = np.zeros_like(bundled)
        result[kept] = 1
        return result

    # Batched result (D, *rest): thin each column over axis 0 independently.
    num_nonzero = ceil(bundled.shape[0] * density)
    flat = bundled.reshape(bundled.shape[0], -1)
    if is_torch:
        out = torch.zeros_like(flat)
        for col in range(flat.shape[1]):
            column = flat[:, col]
            indices = torch.nonzero(column, as_tuple=True)[0]
            if num_nonzero >= indices.numel():
                out[:, col] = column
                continue
            perm = torch.randperm(indices.numel())[:num_nonzero]
            out[indices[perm], col] = 1
        return out.reshape(bundled.shape)

    out = np.zeros_like(flat)
    for col in range(flat.shape[1]):
        column = flat[:, col]
        indices = np.nonzero(column)[0]
        if num_nonzero >= indices.size:
            out[:, col] = column
            continue
        kept = np.random.choice(indices, size=num_nonzero, replace=False)
        out[kept, col] = 1
    return out.reshape(bundled.shape)
