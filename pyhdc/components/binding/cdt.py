from math import log, sqrt
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
# Context-Dependent Thinning
# ============================================================================


def AdditiveContextDependentThinning(
    *hypervectors: ArrayLike,
    probability: Optional[float] = None,
    seed: Optional[int] = None,
) -> ArrayLike:
    """
    Context-dependent thinning for sparse hypervectors.

    Binds sparse vectors while controlling density through selective thinning
    based on context. Used with BSDC encodings.

    Args:
        *hypervectors: Variable number of sparse binary hypervectors, or single 2D batch
        probability: Probability parameter for thinning
        seed: Random seed for reproducibility

    Returns:
        Bound and thinned hypervector

    Note:
        This requires the Disjunction operation from bundling module
    """
    from pyhdc.components.bundling import Disjunction

    hvs, is_torch, _ = _normalize_binding(*hypervectors)
    z = Disjunction(*hypervectors)

    S = hvs[0].shape[0]
    if probability is None:
        probability = 1 / sqrt(S)

    K = int(log(1 - (1 / S)) / log(1 - probability))

    if seed is not None:
        if is_torch:
            torch.manual_seed(seed)
        else:
            np.random.seed(seed)

    if is_torch:
        w = torch.zeros_like(z, dtype=torch.bool)
        for _ in range(1, K + 1):
            r = torch.randint(0, S, (1,)).item()
            if r != 0:
                z_roll = torch.roll(z, shifts=-r, dims=0)
                w = torch.logical_or(w, z_roll)
        result = torch.logical_xor(z.bool(), w).to(z.dtype)
    else:
        w = np.zeros(S, dtype=bool)
        for _ in range(1, K + 1):
            r = np.random.randint(S)
            if r != 0:
                z_roll = np.roll(z, -r)
                w = np.logical_or(w, z_roll)
        result = np.logical_xor(z.astype(bool), w).astype(z.dtype)

    return result
