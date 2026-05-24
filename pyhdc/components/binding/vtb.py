from math import sqrt

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
# Vector-Derived Transformation Binding (VTB)
# ============================================================================


def _vtb_get_y_prime(y: ArrayLike, is_torch: bool) -> ArrayLike:
    """
    Compute the V_y matrix for Vector-Derived Transformation Binding.

    Calculates the block diagonal matrix V_y used in VTB:

    V_y' = d^(1/4) * | y_1       y_2       ...  y_d'  |
                     | y_d'+1    y_d'+2    ...  y_2d' |
                     |  :          :       ...   :    |
                     | y_d-d'+1  y_d-d'+2  ...  y_d   |

    Args:
        y: Input hypervector
        is_torch: Whether using PyTorch backend

    Returns:
        V_y matrix
    """
    size = y.shape[0] if hasattr(y, "shape") else len(y)
    d_prime = int(sqrt(size))
    d_quart = pow(size, 0.25)

    if is_torch:
        # Reshape into square matrix
        V_prime = y.reshape(d_prime, d_prime)
        # Create block diagonal
        Z_prime = torch.zeros((d_prime, d_prime), dtype=y.dtype, device=y.device)
        blocks = [
            [V_prime if i == j else Z_prime for i in range(d_prime)]
            for j in range(d_prime)
        ]
        # Flatten and concatenate blocks
        V_y = torch.block_diag(*[V_prime for _ in range(d_prime)])
        V_y = d_quart * V_y
    else:
        # NumPy implementation
        V_prime = y.reshape(d_prime, d_prime)
        Z_prime = np.zeros([d_prime, d_prime])
        V_y = d_quart * np.block(
            [
                [V_prime if i == j else Z_prime for i in range(d_prime)]
                for j in range(d_prime)
            ]
        )

    return V_y


def VectorDerivedTransformation(*hypervectors: ArrayLike) -> ArrayLike:
    """
    Vector-Derived Transformation Binding.

    Binds vectors using the equation: B_v(y, x) = V_y * x
    where V_y is a matrix derived from vector y.

    For multiple vectors [x1, x2, x3, ..., xn]:
    B_v(x1, x2, x3, ..., xn) = V_xn * V_x(n-1) * ... * V_x2 * x1

    Args:
        *hypervectors: Variable number of hypervectors to bind, or single 2D batch

    Returns:
        Bound hypervector

    Note:
        Requires hypervector dimension to be a perfect fourth power (d = k^4)
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)

    # Compute V_y matrices for all but the first hypervector
    V_y_list = [_vtb_get_y_prime(hvs[i], is_torch) for i in range(1, len(hvs))]
    V_y_list.reverse()  # Reverse for proper multiplication order

    # Multiply matrices together
    if is_torch:
        result = V_y_list[0]
        for i in range(1, len(V_y_list)):
            result = torch.matmul(result, V_y_list[i])
        # Apply to first vector
        x = hvs[0].unsqueeze(1)  # Column vector
        result = torch.matmul(result, x).squeeze()
    else:
        result = V_y_list[0]
        for i in range(1, len(V_y_list)):
            result = np.dot(result, V_y_list[i])
        # Apply to first vector
        x = hvs[0].reshape(-1, 1)  # Column vector
        result = (result @ x).squeeze()

    return result


def TransposeVectorDerivedTransformation(*hypervectors: ArrayLike) -> ArrayLike:
    """
    Pseudo-inverse for Vector-Derived Transformation (unbinding).

    Unbinds vectors using: B+_v(y, x) = V_y^T * x
    where V_y^T is the transpose of the V_y matrix.

    Args:
        *hypervectors: Variable number of hypervectors, or single 2D batch

    Returns:
        Unbound hypervector
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)

    # Compute transposed V_y matrices
    V_y_list = [_vtb_get_y_prime(hvs[i], is_torch).T for i in range(1, len(hvs))]
    V_y_list.reverse()

    # Multiply matrices together
    if is_torch:
        result = V_y_list[0]
        for i in range(1, len(V_y_list)):
            result = torch.matmul(result, V_y_list[i])
        # Apply to first vector
        x = hvs[0].unsqueeze(1)
        result = torch.matmul(result, x).squeeze()
    else:
        result = V_y_list[0]
        for i in range(1, len(V_y_list)):
            result = np.dot(result, V_y_list[i])
        # Apply to first vector
        x = hvs[0].reshape(-1, 1)
        result = (result @ x).squeeze()

    return result
