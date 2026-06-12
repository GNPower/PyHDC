from math import ceil, sqrt
from typing import Optional

import numpy as np

# Optional PyTorch support
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pyhdc.components.input_formatting import _normalize_binding

# Type aliases
from pyhdc.types import ArrayLike

# ============================================================================
# Shift-based Binding
# ============================================================================


def _shift_get_hash(hypervector: ArrayLike, is_torch: bool) -> int:
    """
    Compute hash-based shift amount from hypervector.

    Args:
        hypervector: Binary hypervector to hash
        is_torch: Whether using PyTorch backend

    Returns:
        Shift amount (integer)
    """
    if is_torch:
        indices = torch.nonzero(hypervector, as_tuple=True)[0]
        if len(indices) == 0:
            return 0
        return int(indices.sum().item() % hypervector.shape[0])
    else:
        index_sum = np.sum(np.where(hypervector)[0])
        return int(index_sum % len(hypervector))


def Shifting(*hypervectors: ArrayLike) -> ArrayLike:
    """
    Binding by circular shifting.

    Binds sparse binary hypervectors by rotating positions based on a hash
    of the previous vector. Used with Binary Sparse Distributed Codes (BSDC).

    Args:
        *hypervectors: Variable number of sparse binary hypervectors, or single 2D batch

    Returns:
        Bound hypervector
    """
    hvs, is_torch, _ = _normalize_binding(*hypervectors)

    hashed = _shift_get_hash(hvs[0], is_torch)
    result = hvs[0]

    for i in range(1, len(hvs)):
        if is_torch:
            result = torch.roll(hvs[i], shifts=hashed, dims=0)
        else:
            result = np.roll(hvs[i], hashed)
        hashed = _shift_get_hash(result, is_torch)

    return result


def InverseShifting(*hypervectors: ArrayLike) -> ArrayLike:
    """
    Unbinding operation for shift-based binding.

    Reverses the circular shifts applied during binding.

    Args:
        *hypervectors: Variable number of hypervectors, or single 2D batch

    Returns:
        Unbound hypervector
    """
    hvs, is_torch, _ = _normalize_binding(*hypervectors)

    hashed = _shift_get_hash(hvs[0], is_torch)
    result = hvs[0]

    for i in range(1, len(hvs)):
        if is_torch:
            result = torch.roll(hvs[i], shifts=-hashed, dims=0)
        else:
            result = np.roll(hvs[i], -hashed)
        hashed = _shift_get_hash(result, is_torch)

    return result


# ============================================================================
# Segment Shifting
# ============================================================================


def _segment_shifting_directional(
    *hypervectors: ArrayLike, probability: Optional[float] = None, direction: int = 1
) -> ArrayLike:
    """
    Internal function for segment-based shifting.

    Args:
        *hypervectors: Variable number of hypervectors, or single 2D batch
        probability: Probability parameter for segment size
        direction: 1 for binding, -1 for unbinding

    Returns:
        Shifted hypervector
    """
    hvs, is_torch, _ = _normalize_binding(*hypervectors)
    dimensions = hvs[0].shape[0]

    if probability is None:
        probability = 1 / sqrt(dimensions)

    num_segments = ceil(dimensions * probability)
    seg_size = ceil(dimensions / num_segments)

    result = hvs[0].clone() if is_torch else hvs[0].copy()

    for i in range(1, len(hvs)):
        hv = hvs[i].clone() if is_torch else hvs[i].copy()

        # Process each segment
        for j in range(num_segments - 1):
            seg_start = j * seg_size
            seg_end = seg_start + seg_size
            segment = result[seg_start:seg_end]

            # Compute roll index from segment
            if is_torch:
                powers = torch.arange(
                    segment.shape[0] - 1, -1, -1, device=segment.device
                )
                roll_index = int((segment * (2**powers)).sum().item() * direction)
                hv[seg_start:seg_end] = torch.roll(
                    hv[seg_start:seg_end], shifts=roll_index, dims=0
                )
            else:
                powers = np.arange(segment.shape[0] - 1, -1, -1)
                roll_index = int(segment.dot(2**powers) * direction)
                hv[seg_start:seg_end] = np.roll(hv[seg_start:seg_end], roll_index)

        # Process final segment (may be smaller)
        seg_start = (num_segments - 1) * seg_size
        segment = result[seg_start:]

        if is_torch:
            powers = torch.arange(segment.shape[0] - 1, -1, -1, device=segment.device)
            roll_index = int((segment * (2**powers)).sum().item() * direction)
            hv[seg_start:] = torch.roll(hv[seg_start:], shifts=roll_index, dims=0)
        else:
            powers = np.arange(segment.shape[0] - 1, -1, -1)
            roll_index = int(segment.dot(2**powers) * direction)
            hv[seg_start:] = np.roll(hv[seg_start:], roll_index)

        result = hv

    return result


def SegmentShifting(
    *hypervectors: ArrayLike, probability: Optional[float] = None
) -> ArrayLike:
    """
    Segment-based circular shifting for binding.

    Divides hypervectors into segments and applies different rotations to each
    segment based on the content of previous vectors.

    Args:
        *hypervectors: Variable number of sparse binary hypervectors, or single 2D batch
        probability: Probability parameter controlling number of segments

    Returns:
        Bound hypervector
    """
    return _segment_shifting_directional(
        *hypervectors, probability=probability, direction=1
    )


def InverseSegmentShifting(
    *hypervectors: ArrayLike, probability: Optional[float] = None
) -> ArrayLike:
    """
    Unbinding operation for segment-based shifting.

    Args:
        *hypervectors: Variable number of hypervectors, or single 2D batch
        probability: Probability parameter (must match binding)

    Returns:
        Unbound hypervector
    """
    return _segment_shifting_directional(
        *hypervectors, probability=probability, direction=-1
    )
