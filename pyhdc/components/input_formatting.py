from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np

# Optional PyTorch support
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.types import ArrayLike

# Avoid circular imports
if TYPE_CHECKING:
    from pyhdc.hypervector import Hypervector


def _is_hypervector(obj: Any) -> bool:
    """Check if an object is a Hypervector without importing the class."""
    return hasattr(obj, "data") and hasattr(obj, "encoding") and hasattr(obj, "backend")


def _extract_data(obj: Union[ArrayLike, "Hypervector"]) -> ArrayLike:
    """
    Extract raw data from a single Hypervector or return array as-is.

    Simple helper for single-argument cases like thin() and similarity().

    Args:
        obj: Hypervector object or raw array

    Returns:
        Raw array data
    """
    return obj.data if _is_hypervector(obj) else obj


def _normalize_inputs(
    *arrays: Union[ArrayLike, "Hypervector"]
) -> Tuple[List[ArrayLike], bool, Optional["Hypervector"]]:
    """
    Extract raw data from Hypervectors and normalize to list of arrays.

    Handles these cases:
    1. Multiple Hypervector objects: func(hv1, hv2, hv3, ...) â†’ extract all .data
    2. Multiple raw arrays: func(arr1, arr2, arr3, ...) â†’ return as-is
    3. Single Hypervector or array: func(hv) â†’ extract .data if needed

    Args:
        *arrays: Variable number of arrays or Hypervector objects

    Returns:
        Tuple of (list of raw arrays, is_torch flag, reference_hypervector)
        - data_arrays: List of raw arrays (extracted from Hypervectors)
        - is_torch: Whether using PyTorch backend
        - reference_hv: First Hypervector encountered (for encoding/backend info)
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required")

    # Extract data from all inputs
    data_arrays = []
    reference_hv = None
    is_torch = False

    for i, item in enumerate(arrays):
        if _is_hypervector(item):
            # Store first Hypervector as reference
            if reference_hv is None:
                reference_hv = item
                is_torch = item.backend == "torch"
            data_arrays.append(item.data)
        else:
            # Raw array
            if i == 0 and not reference_hv:
                is_torch = TORCH_AVAILABLE and torch.is_tensor(item)
            data_arrays.append(item)

    return data_arrays, is_torch, reference_hv


def _detect_batch_structure(
    *arrays: Union[ArrayLike, "Hypervector", List], batch_dim: Optional[int] = None
) -> Tuple[bool, Union[List, List[List]]]:
    """
    Detect if input represents batched operations for bundle/bind/unbind.

    Batching Rules:
    1. Explicit batch_dim with array â†’ Split array along batch_dim â†’ Batched
    2. Nested list [[...], [...]] â†’ Each inner list is a batch â†’ Batched
    3. Flat list [...] â†’ Bundle all together â†’ NOT batched
    4. Multiple args (arg1, arg2, ...) â†’ Bundle all together â†’ NOT batched

    Args:
        *arrays: Variable number of arrays, Hypervectors, or lists
        batch_dim: Dimension along which to split arrays for batching

    Returns:
        Tuple of (is_batched, groups):
        - is_batched: True if batched operation
        - groups: If batched, list of batch groups; otherwise original input

    Examples:
        >>> _detect_batch_structure([[hv1, hv2], [hv3, hv4]])
        (True, [[hv1, hv2], [hv3, hv4]])

        >>> _detect_batch_structure([hv1, hv2, hv3])
        (False, [hv1, hv2, hv3])

        >>> _detect_batch_structure(hv1, hv2, hv3)
        (False, (hv1, hv2, hv3))
    """
    # Case 1: Explicit batch_dim with array
    if batch_dim is not None:
        if len(arrays) != 1:
            raise ValueError("batch_dim parameter only valid with single array input")

        arr = arrays[0]

        # Check if it's an array with ndim
        if not hasattr(arr, "ndim") and not hasattr(arr, "shape"):
            raise ValueError(f"batch_dim requires array input, got {type(arr)}")

        ndim = arr.ndim if hasattr(arr, "ndim") else len(arr.shape)
        if ndim < 3:
            raise ValueError(f"batch_dim requires 3D+ array, got {ndim}D")

        if batch_dim < 0 or batch_dim >= ndim:
            raise ValueError(f"batch_dim {batch_dim} out of range for {ndim}D array")

        # Split array along batch dimension
        if TORCH_AVAILABLE and torch.is_tensor(arr):
            # PyTorch: split along batch_dim
            batch_groups = [arr[i] for i in range(arr.shape[batch_dim])]
        else:
            # NumPy: split along batch_dim
            batch_groups = [
                np.take(arr, i, axis=batch_dim) for i in range(arr.shape[batch_dim])
            ]

        return (True, batch_groups)

    # Case 2: Single list input - check if nested
    if len(arrays) == 1 and isinstance(arrays[0], list):
        lst = arrays[0]

        # Empty list
        if len(lst) == 0:
            return (False, arrays)

        # Check if nested list (first element is also a list)
        if isinstance(lst[0], list):
            # Nested list â†’ batched
            return (True, lst)
        else:
            # Flat list â†’ not batched (bundle together)
            return (False, lst)

    # Case 3: Multiple arguments or single non-list â†’ not batched
    if len(arrays) == 1:
        return (False, arrays)
    else:
        return (False, arrays)


def _wrap_as_hypervector(
    data: ArrayLike, reference_hv: Optional["Hypervector"]
) -> Union[ArrayLike, "Hypervector"]:
    """
    Wrap array data as a Hypervector if a reference Hypervector was provided.

    Args:
        data: The array data to potentially wrap
        reference_hv: Reference Hypervector to get encoding and backend from

    Returns:
        Either the raw data (if no reference) or a new Hypervector
    """
    if reference_hv is None:
        return data

    # Import here to avoid circular dependency
    from pyhdc.hypervector import Hypervector

    return Hypervector(data, reference_hv.encoding, reference_hv.backend)


def _get_backend_ops(is_torch: bool):
    """Get backend-specific operations."""
    if is_torch:
        return torch
    return np
