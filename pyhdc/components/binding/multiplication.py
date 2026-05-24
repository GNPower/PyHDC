from typing import List, Optional, Tuple

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
# Multiplication-based Binding
# ============================================================================


def ElementMultiplication(*hypervectors: ArrayLike) -> ArrayLike:
    """
    Element-wise multiplication binding.

    Binds hypervectors by multiplying corresponding elements. Commonly used
    with MAP encodings (bipolar values).

    Args:
        *hypervectors: Variable number of hypervectors to bind, or single 2D batch

    Returns:
        Bound hypervector

    Example:
        >>> v1 = np.array([1, -1, 1, -1])
        >>> v2 = np.array([1, 1, -1, -1])
        >>> result = ElementMultiplication(v1, v2)
        >>> # result: [1, -1, -1, 1]
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)

    if is_torch:
        return torch.prod(torch.stack(hvs), dim=0)
    else:
        return np.multiply.reduce(hvs)


# ============================================================================
# Matrix-based Binding
# ============================================================================


def MatrixMultiplication(
    *hypervectors: ArrayLike,
    matrices: Optional[List[ArrayLike]] = None,
    seed: Optional[int] = None,
) -> Tuple[ArrayLike, List[ArrayLike]]:
    """
    Matrix-based binding using random orthogonal matrices.

    Binds hypervectors by transforming them with random matrices. Returns both
    the bound result and the matrices for later unbinding.

    Args:
        *hypervectors: Variable number of hypervectors to bind, or single 2D batch
        matrices: Optional pre-generated matrices. If None, generates new ones.
        seed: Random seed for reproducible matrix generation

    Returns:
        Tuple of (bound hypervector, list of matrices)

    Note:
        Requires N-1 matrices to bind N hypervectors
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)
    dim = hvs[0].shape[0]

    # Generate or validate matrices
    if matrices is None:
        if seed is not None:
            if is_torch:
                torch.manual_seed(seed)
            else:
                np.random.seed(seed)

        if is_torch:
            matrices = [
                torch.randint(
                    0, 2, (dim, dim), dtype=hvs[0].dtype, device=hvs[0].device
                )
                for _ in range(len(hvs) - 1)
            ]
        else:
            matrices = [
                np.random.randint(0, 2, size=(dim, dim)).astype(hvs[0].dtype)
                for _ in range(len(hvs) - 1)
            ]
    else:
        if len(matrices) != len(hvs) - 1:
            raise ValueError(
                f"Requires {len(hvs) - 1} matrices for {len(hvs)} hypervectors, "
                f"got {len(matrices)}"
            )

    # Perform binding
    result = hvs[0]
    for i in range(len(hvs) - 1):
        if is_torch:
            result = torch.matmul(result + hvs[i + 1], matrices[i])
        else:
            result = np.matmul(result + hvs[i + 1], matrices[i])

    return result, matrices


def InverseMatrixMultiplication(
    *hypervectors: ArrayLike, matrices: List[ArrayLike]
) -> ArrayLike:
    """
    Unbinding operation for matrix-based binding.

    Recovers the original hypervector using the stored matrices.

    Args:
        *hypervectors: Variable number of hypervectors (bound result + unbinding keys)
        matrices: List of matrices used during binding

    Returns:
        Unbound hypervector
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)

    if len(matrices) != len(hvs) - 1:
        raise ValueError(
            f"Requires {len(hvs) - 1} matrices for {len(hvs)} hypervectors, "
            f"got {len(matrices)}"
        )

    result = hvs[0]
    for i in range(len(hvs) - 1):
        if is_torch:
            result = torch.matmul(result, matrices[i].T) - hvs[i + 1]
        else:
            result = np.matmul(result, matrices[i].T) - hvs[i + 1]

    return result
