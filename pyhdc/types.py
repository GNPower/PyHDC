# Type aliases
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple, Union

import numpy as np

# Optional PyTorch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Avoid circular imports
if TYPE_CHECKING:
    from pyhdc.hypervector import Hypervector

Backend = Literal["numpy", "torch"]
ArrayLike = Union[
    np.ndarray, "torch.Tensor"
]  # pyright: ignore[reportOptionalMemberAccess]
HypervectorLike = Union["Hypervector", List["Hypervector"]]
SequenceType = Union[List[int], List[float], ArrayLike]
Device = Union[str, "torch.device"]  # pyright: ignore[reportOptionalMemberAccess]
GeneratorOutputType = Literal["bits", "words", "floats"]

# Batch operation types
HypervectorOrList = Union["Hypervector", List["Hypervector"]]
HypervectorInput = Union[
    ArrayLike, "Hypervector", List["Hypervector"], List[List["Hypervector"]]
]
SimilarityResult = Union[float, ArrayLike, List[Union[float, ArrayLike]]]

# Metadata types for HDC operations
OperationMetadata = Dict[str, Any]
OperationResult = Union[ArrayLike, Tuple[ArrayLike, OperationMetadata]]
