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
    # Order matters: probe ``encoding``/``backend`` before ``data``. A raw array
    # has no ``encoding``, so the check evaluates to False before ``data`` is
    # ever checked. Checking ``data`` on some raw arrays raises exceptions (numpy
    # raises ``ValueError: cannot include dtype 'E' in a buffer`` for ml_dtypes
    # such as bfloat16, whose char kind is ``'E'``), and ``hasattr`` only catches
    # ``AttributeError``, so that ValueError would otherwise propagate and crash.
    return hasattr(obj, "encoding") and hasattr(obj, "backend") and hasattr(obj, "data")


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


def _resolve_reduce_axes(
    input_ndim: int, axis: Union[None, int, Tuple[int, ...]]
) -> Tuple[int, ...]:
    """
    Resolve a user ``axis`` spec to a validated tuple of reduce axes.

    ``axis=None`` selects the last batch axis (so ``(D, N)`` reduces axis 1 as in
    2.0 and ``(D, N, M)`` reduces axis 2). Negatives are normalized. Duplicates,
    out-of-range axes, and axis 0 (the hypervector dimension ``D``) are rejected.
    """
    if axis is None:
        if input_ndim <= 1:
            return ()
        return (input_ndim - 1,)

    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    resolved: List[int] = []
    for value in axes:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"axis must be an int or tuple of ints, got {axis!r}")
        norm = int(value) + input_ndim if value < 0 else int(value)
        if norm < 0 or norm >= input_ndim:
            raise ValueError(f"axis {value} is out of range for a {input_ndim}D array")
        if norm == 0:
            raise ValueError(
                "axis 0 is the hypervector dimension and cannot be reduced"
            )
        resolved.append(norm)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"duplicate reduce axes in {axis!r}")
    return tuple(resolved)


def _reduce_count(batch: ArrayLike, reduce_axes: Tuple[int, ...]) -> int:
    """Number of hypervectors folded together (product of the reduce-axis sizes)."""
    count = 1
    for a in reduce_axes:
        count *= int(batch.shape[a])
    return count


def _zone_count(mask: ArrayLike, is_torch: bool) -> Any:
    """
    Random-zone count for bundling metadata, gated on the result rank.

    Returns a Python ``int`` for a ``(D,)`` result (the 2.0 contract) and a
    per-output-vector count array (summed over axis 0) for a surviving-batch
    result such as ``(D, M)``.
    """
    if getattr(mask, "ndim", 1) == 1:
        return int(mask.sum().item()) if is_torch else int(mask.sum())
    return mask.sum(dim=0) if is_torch else mask.sum(axis=0)


def _pad_trailing(array: ArrayLike, target_ndim: int, is_torch: bool) -> ArrayLike:
    """Append trailing length-1 axes until ``array`` has ``target_ndim`` dims."""
    del is_torch  # numpy and torch both support ``array[..., None]``
    while getattr(array, "ndim", 1) < target_ndim:
        array = array[..., None]
    return array


def _require_single_vector(operands: List[ArrayLike], op_name: str) -> None:
    """Raise if any operand is batched (ndim > 1) for a non-batch-safe operation."""
    for operand in operands:
        if getattr(operand, "ndim", 1) > 1:
            raise ValueError(
                f"{op_name} only supports single (D,) hypervectors; got shape "
                f"{tuple(operand.shape)}. Use batch_dim= at the Encoding layer "
                f"to loop over a batch."
            )


def _broadcast_operands(operands: List[ArrayLike], is_torch: bool) -> List[ArrayLike]:
    """
    Pad operands with trailing length-1 axes so they broadcast element-wise.

    Aligns mixed ranks for element-wise binding: a ``(D,)`` operand bound against a
    ``(D, N)`` batch is padded to ``(D, 1)`` so it broadcasts over the N columns.
    Equal-rank operands are returned unchanged (2.0 behavior).
    """
    target = max(getattr(op, "ndim", 1) for op in operands)
    return [_pad_trailing(op, target, is_torch) for op in operands]


def _normalize_bundling(
    *arrays: Union[ArrayLike, "Hypervector"],
    axis: Union[None, int, Tuple[int, ...]] = None,
) -> Tuple[ArrayLike, bool, Optional["Hypervector"], Tuple[int, ...]]:
    """
    Normalize bundling inputs to a dimension-first batch plus its reduce axes.

    Axis 0 is always the hypervector dimension ``D``; the folded axis(es) are the
    batch axes:

    - ``bundle(a, b, c)`` with ``(D,)`` vectors      -> ``(D, 3)``, reduce ``(1,)``
    - ``bundle(batch)`` with a ``(D, N)`` batch      -> ``(D, N)``, reduce ``(1,)``
    - ``bundle(a, batch)`` mixing the two            -> ``(D, 1 + N)``, reduce ``(1,)``
    - ``bundle(tensor)`` with a ``(D, N, M)`` batch  -> kept unflattened, reduce the
      last batch axis ``(2,)`` by default, or an explicit ``axis``

    ``axis`` selects which batch axis(es) collapse (never axis 0). A lone ``(D,)``
    input accepts only ``axis=None``. Multiple operands must be ``(D,)`` or
    ``(D, N)`` (an operand with ``ndim >= 3`` raises).

    Returns:
        Tuple of ``(batch, is_torch, reference_hypervector, reduce_axes)``.
    """
    data_arrays, is_torch, reference_hv = _normalize_inputs(*arrays)
    ndims = [getattr(arr, "ndim", 1) for arr in data_arrays]
    multi = len(data_arrays) > 1

    if multi and any(nd >= 3 for nd in ndims):
        raise ValueError(
            "bundling multiple operands requires (D,) or (D, N) inputs; "
            "got an operand with ndim >= 3"
        )

    single_1d = not multi and ndims[0] == 1

    if not multi and ndims[0] >= 3:
        batch = data_arrays[0]
    else:
        columns = [_as_2d_columns(arr, is_torch) for arr in data_arrays]
        if len(columns) == 1:
            batch = columns[0]
        elif is_torch:
            batch = torch.cat(columns, dim=1)
        else:
            batch = np.concatenate(columns, axis=1)

    if single_1d:
        if axis is not None:
            raise ValueError("axis is not valid for a single (D,) hypervector")
        reduce_axes: Tuple[int, ...] = (1,)
    else:
        reduce_axes = _resolve_reduce_axes(batch.ndim, axis)

    return batch, is_torch, reference_hv, reduce_axes


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
    *arrays: Union[ArrayLike, "Hypervector"],
    axis: Optional[int] = None,
    mode: str = "pairwise",
) -> Tuple[ArrayLike, ArrayLike, bool, bool]:
    """
    Normalize similarity inputs to an aligned ``(A, B)`` pair reduced over axis 0.

    Reduction is always over axis 0 (the dimension ``D``); the trailing batch axes
    are aligned by broadcasting:

    - one ``(D, N)`` batch (``axis=None``)  -> ``A = col 0``, ``B = cols 1..N-1``
    - one ``(D, N, M, ...)`` batch          -> requires an explicit ``axis``; splits
      index 0 vs the rest along that axis (the split axis is kept for broadcasting)
    - two ``(D,)`` vectors                  -> one scalar score (``scalar=True``)
    - two ``(D, N)`` batches                -> N per-column scores
    - mixed ranks (e.g. ``(D,)`` vs ``(D, N, M)``) -> lower-rank operand padded with
      trailing length-1 axes, then broadcast

    With ``mode="cross"`` the two inputs are returned unaligned as the raw ``(D, P)``
    and ``(D, M)`` operands (no column split, no padding) for the full outer-product
    cross-similarity path, ``scalar`` is always False.

    Returns:
        Tuple of ``(A, B, is_torch, scalar)``. ``scalar`` is True only when both
        inputs were a single 1D vector (return the result as a Python float).
    """
    if mode == "cross":
        if axis is not None:
            raise ValueError('similarity mode="cross" does not accept axis=')
        data_arrays, is_torch, _ = _normalize_inputs(*arrays)
        if len(data_arrays) != 2:
            raise ValueError(
                'similarity mode="cross" requires exactly two (D, P) and (D, M) inputs'
            )
        a, b = data_arrays[0], data_arrays[1]
        if getattr(a, "ndim", 1) != 2 or getattr(b, "ndim", 1) != 2:
            raise ValueError(
                'similarity mode="cross" requires 2D (D, P) and (D, M) batches'
            )
        if a.shape[0] != b.shape[0]:
            raise ValueError(
                f"cross similarity requires matching dimension D, "
                f"got {a.shape[0]} and {b.shape[0]}"
            )
        return a, b, is_torch, False

    data_arrays, is_torch, _ = _normalize_inputs(*arrays)

    if len(data_arrays) == 1:
        arr = data_arrays[0]
        if getattr(arr, "ndim", 1) < 2:
            raise ValueError(
                "single-input similarity requires a 2D (D, N) batch of hypervectors"
            )
        if axis is None:
            if arr.ndim >= 3:
                raise ValueError(
                    "single-input similarity on a (D, N, M, ...) batch requires an "
                    "explicit axis"
                )
            split_axis = 1
        else:
            resolved = _resolve_reduce_axes(arr.ndim, axis)
            if len(resolved) != 1:
                raise ValueError(
                    "single-input similarity reduces exactly one batch axis"
                )
            split_axis = resolved[0]

        size = arr.shape[split_axis]
        if is_torch:
            head = torch.tensor([0], device=arr.device)
            rest = torch.arange(1, size, device=arr.device)
            return (
                arr.index_select(split_axis, head),
                arr.index_select(split_axis, rest),
                is_torch,
                False,
            )
        return (
            np.take(arr, [0], axis=split_axis),
            np.take(arr, list(range(1, size)), axis=split_axis),
            is_torch,
            False,
        )

    if len(data_arrays) == 2:
        a, b = data_arrays[0], data_arrays[1]
        scalar = getattr(a, "ndim", 1) == 1 and getattr(b, "ndim", 1) == 1
        target = 2 if scalar else max(a.ndim, b.ndim)
        a = _pad_trailing(a, target, is_torch)
        b = _pad_trailing(b, target, is_torch)
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

    # Case 2: Single list input, check if nested
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

    # Case 3: Multiple arguments or single non-list, not batched
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
