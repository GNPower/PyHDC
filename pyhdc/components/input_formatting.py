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


def _as_2d_columns(array: ArrayLike, is_torch: bool) -> ArrayLike:
    """
    Reshape a hypervector array so its columns are individual hypervectors.

    A 1D ``(D,)`` vector becomes ``(D, 1)``; a higher-rank ``(D, *batch)`` array is
    flattened to ``(D, prod(batch))``. A 2D ``(D, N)`` array is returned unchanged.
    """
    ndim = getattr(array, "ndim", 1)
    if ndim == 1:
        return array[:, None]
    if ndim > 2:
        return array.reshape(array.shape[0], -1)
    return array


def _normalize_bundling(
    *arrays: Union[ArrayLike, "Hypervector"]
) -> Tuple[ArrayLike, bool, Optional["Hypervector"]]:
    """
    Normalize bundling inputs to a single ``(D, N)`` array of column hypervectors.

    Every supported call shape collapses to one dimension-first 2D array whose
    columns are the hypervectors to bundle, so each bundling function only has to
    reduce over axis 1:

    - ``bundle(a, b, c)`` with ``(D,)`` vectors      -> ``(D, 3)``
    - ``bundle(batch)`` with a ``(D, N)`` batch      -> ``(D, N)``
    - ``bundle(a, batch)`` mixing the two            -> concatenated ``(D, 1 + N)``

    Args:
        *arrays: Hypervectors or raw arrays to bundle together.

    Returns:
        Tuple of ``(batch, is_torch, reference_hypervector)`` where ``batch`` is a
        single ``(D, N)`` array.
    """
    data_arrays, is_torch, reference_hv = _normalize_inputs(*arrays)
    columns = [_as_2d_columns(arr, is_torch) for arr in data_arrays]

    if len(columns) == 1:
        batch = columns[0]
    elif is_torch:
        batch = torch.cat(columns, dim=1)
    else:
        batch = np.concatenate(columns, axis=1)

    return batch, is_torch, reference_hv


def _normalize_binding(
    *arrays: Union[ArrayLike, "Hypervector"]
) -> Tuple[List[ArrayLike], bool, Optional["Hypervector"]]:
    """
    Normalize binding inputs to a list of operands for element-wise reduction.

    Binding combines K operands position-by-position, so each operand keeps its
    own shape (``(D,)`` or ``(D, N)``); two equal-shaped ``(D, N)`` batches bind
    per column (column i of operand 1 with column i of operand 2).

    Args:
        *arrays: Hypervectors or raw arrays to bind.

    Returns:
        Tuple of ``(operands, is_torch, reference_hypervector)``.
    """
    return _normalize_inputs(*arrays)


def _normalize_similarity(
    *arrays: Union[ArrayLike, "Hypervector"]
) -> Tuple[ArrayLike, ArrayLike, bool, bool]:
    """
    Normalize similarity inputs to an aligned ``(A, B)`` pair compared column-wise.

    Resolves every calling convention to two dimension-first arrays whose columns
    are compared over axis 0 (broadcasting on axis 1):

    - one ``(D, N)`` batch         -> ``A = col 0``, ``B = cols 1..N-1`` (N-1 scores)
    - two ``(D,)`` vectors         -> one scalar score (``scalar=True``)
    - two ``(D, N)`` batches       -> N per-column scores
    - ``(D,)`` and ``(D, N)``      -> the vector broadcast against each column (N scores)

    Args:
        *arrays: One or two hypervectors / raw arrays.

    Returns:
        Tuple of ``(A, B, is_torch, scalar)``. ``A``/``B`` are dimension-first
        arrays broadcastable on axis 1; ``scalar`` is True when both inputs were a
        single 1D vector (the result should be returned as a Python float).
    """
    data_arrays, is_torch, _ = _normalize_inputs(*arrays)

    if len(data_arrays) == 1:
        arr = data_arrays[0]
        if getattr(arr, "ndim", 1) < 2:
            raise ValueError(
                "single-input similarity requires a 2D (D, N) batch of hypervectors"
            )
        return arr[:, :1], arr[:, 1:], is_torch, False

    if len(data_arrays) == 2:
        a, b = data_arrays[0], data_arrays[1]
        a_1d = getattr(a, "ndim", 1) == 1
        b_1d = getattr(b, "ndim", 1) == 1
        scalar = a_1d and b_1d
        if a_1d:
            a = a[:, None]
        if b_1d:
            b = b[:, None]
        return a, b, is_torch, scalar

    raise ValueError(
        f"similarity expects one or two hypervector inputs, got {len(data_arrays)}"
    )


def _normalize_thinning(
    array: Union[ArrayLike, "Hypervector"],
) -> Tuple[ArrayLike, bool, Optional["Hypervector"]]:
    """
    Normalize a single thinning input to its raw array.

    Thinning acts on one hypervector (``(D,)``) or, where the thinning function
    supports it, per column of a ``(D, N)`` batch.

    Args:
        array: A hypervector or raw array.

    Returns:
        Tuple of ``(data, is_torch, reference_hypervector)``.
    """
    data = _extract_data(array)
    is_torch = TORCH_AVAILABLE and torch.is_tensor(data)
    reference_hv = array if _is_hypervector(array) else None
    return data, is_torch, reference_hv


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
