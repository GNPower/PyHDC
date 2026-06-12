#!/usr/bin/env python
"""
Hyperdimensional Computing Library

A professional library for hyperdimensional computing with support for
multiple backends (NumPy and PyTorch), custom generators, and recovery methods.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from pyhdc.types import ArrayLike, Backend, Device, GeneratorOutputType

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from pyhdc.encodings.base import Encoding

# Optional PyTorch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class EncodingSpec:
    """Specification for a hypervector encoding scheme."""

    dtype: Any
    element_generator: Callable
    similarity_fn: Callable
    bundling_fn: Callable
    thinning_fn: Callable
    binding_fn: Callable
    unbinding_fn: Callable
    mask: Optional[int] = None
    generator_output_type: GeneratorOutputType = "floats"  # What type of output needed


class BackendManager:
    """Manages backend operations for numpy and PyTorch."""

    @staticmethod
    def get_backend(array: ArrayLike) -> Backend:
        """Determine the backend of an array."""
        if TORCH_AVAILABLE and torch.is_tensor(array):
            return "torch"
        return "numpy"

    @staticmethod
    def to_numpy(array: ArrayLike) -> np.ndarray:
        """Convert array to numpy."""
        if TORCH_AVAILABLE and torch.is_tensor(array):
            return array.detach().cpu().numpy()
        return np.asarray(array)

    @staticmethod
    def to_torch(
        array: ArrayLike, device: Optional[Device] = None
    ) -> "torch.Tensor":  # pyright: ignore[reportInvalidTypeForm]
        """Convert array to PyTorch tensor."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install it with: pip install torch"
            )

        if torch.is_tensor(array):
            return array.to(device) if device else array

        tensor = torch.from_numpy(np.asarray(array))
        return tensor.to(device) if device else tensor

    @staticmethod
    def get_device(array: ArrayLike) -> Optional[Device]:
        """Get the device of a tensor (None for numpy arrays)."""
        if TORCH_AVAILABLE and torch.is_tensor(array):
            return array.device
        return None


class Hypervector:
    """
    A hypervector representation supporting multiple backends.

    Similar to numpy's ndarray, this class can represent a single hypervector
    or an array of hypervectors, and supports both numpy and PyTorch backends.

    Attributes:
        data: The underlying array (numpy.ndarray or torch.Tensor)
        encoding: The encoding scheme used for operations
        backend: The backend being used ('numpy' or 'torch')
    """

    def __init__(
        self,
        data: ArrayLike,
        encoding: "Encoding",
        backend: Optional[Backend] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a Hypervector.

        Args:
            data: The underlying array data
            encoding: The encoding scheme for this hypervector
            backend: Backend to use (auto-detected if None)
            metadata: Optional operational metadata dict
        """
        self._encoding = encoding

        if backend is None:
            backend = BackendManager.get_backend(data)

        self._backend = backend
        self._data = data
        self._metadata = metadata if metadata is not None else {}

    @property
    def data(self) -> ArrayLike:
        """Get the underlying array data."""
        return self._data

    @property
    def encoding(self) -> "Encoding":
        """Get the encoding scheme."""
        return self._encoding

    @property
    def backend(self) -> Backend:
        """Get the current backend."""
        return self._backend

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the hypervector."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return self._data.ndim

    @property
    def dtype(self) -> Any:
        """Get the data type."""
        return self._data.dtype

    @property
    def device(self) -> Optional[Device]:
        """Get the device (for PyTorch backend)."""
        return BackendManager.get_device(self._data)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get operational metadata for this hypervector.

        Returns:
            Dictionary containing metadata from the operation that created
            this hypervector. Empty dict if no metadata available.
        """
        return self._metadata.copy()  # Return copy to prevent mutation

    def __repr__(self) -> str:
        return f"Hypervector(shape={self.shape}, backend='{self.backend}', encoding={self.encoding.__class__.__name__})"

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key) -> "Hypervector":
        """Support indexing and slicing."""
        return Hypervector(
            self._data[key], self._encoding, self._backend, self._metadata
        )

    def select(self, indices) -> "Hypervector":
        """
        Select hypervectors along the batch axis (axis 1).

        Hypervectors are dimension-first ``(D, N)``; ``select`` keeps the columns
        at the given indices.

        Args:
            indices: Integer (non-negative) indices of the hypervectors to keep,
                as a sequence, numpy array, or tensor.

        Returns:
            A new Hypervector of shape ``(D, len(indices))`` with the selected
            columns, preserving encoding, backend, and metadata.
        """
        data = self._data
        if self._backend == "torch":
            idx = (
                indices
                if torch.is_tensor(indices)
                else torch.as_tensor(np.asarray(indices))
            )
            idx = idx.to(device=data.device, dtype=torch.long)
            selected = data.index_select(1, idx)
        else:
            selected = data[:, np.asarray(indices, dtype=np.intp)]
        return Hypervector(selected, self._encoding, self._backend, self._metadata)

    def to_numpy(self) -> "Hypervector":
        """Convert to numpy backend."""
        if self._backend == "numpy":
            return self
        return Hypervector(
            BackendManager.to_numpy(self._data), self._encoding, "numpy", self._metadata
        )

    def to_torch(self, device: Optional[Device] = None) -> "Hypervector":
        """Convert to PyTorch backend."""
        if self._backend == "torch" and device is None:
            return self
        return Hypervector(
            BackendManager.to_torch(self._data, device),
            self._encoding,
            "torch",
            self._metadata,
        )

    def to(self, device: Device) -> "Hypervector":
        """Move to specified device (PyTorch only)."""
        if self._backend != "torch":
            raise ValueError("to() method is only available for PyTorch backend")
        return self.to_torch(device)

    def cpu(self) -> "Hypervector":
        """Move to CPU."""
        if self._backend == "torch":
            return self.to("cpu")
        return self

    def cuda(self, device: Optional[int] = None) -> "Hypervector":
        """Move to CUDA device."""
        device_str = f"cuda:{device}" if device is not None else "cuda"
        return self.to_torch(device_str)

    def similarity(
        self, other: Optional["Hypervector"] = None
    ) -> Union[
        float, np.ndarray, "torch.Tensor"
    ]:  # pyright: ignore[reportInvalidTypeForm]
        """
        Compute similarity with another hypervector, or within a batch.

        Args:
            other: Another hypervector to compare with. If omitted, ``self`` must be
                a ``(D, N)`` batch and the similarity of column 0 against each
                remaining column is returned.

        Returns:
            Similarity score(s)
        """
        if other is None:
            return self._encoding.similarity(self)
        self._check_compatibility(other)
        return self._encoding.similarity(self, other)

    def bundle(
        self, *others: "Hypervector", batch_dim: Optional[int] = None
    ) -> "Hypervector":
        """
        Bundle this hypervector with others.

        Note: Batching via batch_dim is available at the Encoding class level,
        not at the instance method level (which always operates on single instances).

        Args:
            *others: Other hypervectors to bundle
            batch_dim: Passed through to Encoding.bundle (instance methods don't use batching)

        Returns:
            A new bundled hypervector
        """
        for other in others:
            self._check_compatibility(other)

        # Encoding.bundle now handles Hypervector objects and returns a Hypervector
        return self._encoding.bundle(self, *others, batch_dim=batch_dim)

    def bind(
        self, *others: "Hypervector", batch_dim: Optional[int] = None
    ) -> "Hypervector":
        """
        Bind this hypervector with others.

        Note: Batching via batch_dim is available at the Encoding class level,
        not at the instance method level (which always operates on single instances).

        Args:
            *others: Other hypervectors to bind
            batch_dim: Passed through to Encoding.bind (instance methods don't use batching)

        Returns:
            A new bound hypervector
        """
        for other in others:
            self._check_compatibility(other)

        # Encoding.bind now handles Hypervector objects and returns a Hypervector
        return self._encoding.bind(self, *others, batch_dim=batch_dim)

    def unbind(
        self, *others: "Hypervector", batch_dim: Optional[int] = None
    ) -> "Hypervector":
        """
        Unbind this hypervector from others.

        Note: Batching via batch_dim is available at the Encoding class level,
        not at the instance method level (which always operates on single instances).

        Args:
            *others: Other hypervectors to unbind
            batch_dim: Passed through to Encoding.unbind (instance methods don't use batching)

        Returns:
            A new unbound hypervector
        """
        for other in others:
            self._check_compatibility(other)

        # Encoding.unbind now handles Hypervector objects and returns a Hypervector
        return self._encoding.unbind(self, *others, batch_dim=batch_dim)

    def thin(self) -> "Hypervector":
        """
        Apply thinning operation.

        Returns:
            A new thinned hypervector
        """
        # Encoding.thin now handles Hypervector objects and returns a Hypervector
        return self._encoding.thin(self)

    def _check_compatibility(self, other: "Hypervector") -> None:
        """Check if another hypervector is compatible."""
        if self._backend != other._backend:
            raise ValueError(
                f"Backend mismatch: {self._backend} vs {other._backend}. "
                f"Use .to_numpy() or .to_torch() to convert."
            )
        if self._encoding.__class__ != other._encoding.__class__:
            warnings.warn(
                f"Encoding mismatch: {self._encoding.__class__.__name__} vs "
                f"{other._encoding.__class__.__name__}"
            )


# Convenience functions for API


def generate(
    encoding: Encoding,
    size: Union[int, Tuple[int, ...]],
    use_generator: Optional[bool] = None,
) -> Hypervector:
    """
    Generate random hypervector(s) using the specified encoding.

    Args:
        encoding: The encoding scheme to use
        size: Size of hypervector(s) to generate
        use_generator: Whether to use the custom generator

    Returns:
        A new Hypervector
    """
    return encoding.generate(size, use_generator=use_generator)


def zeros(encoding: Encoding, size: Union[int, Tuple[int, ...]] = None) -> Hypervector:
    """
    Generate zero hypervector(s) using the specified encoding.

    Args:
        encoding: The encoding scheme to use
        size: Size of hypervector(s) to generate

    Returns:
        A new zero Hypervector
    """
    return encoding.zeros(size)


def bundle(*hypervectors: Hypervector) -> Hypervector:
    """
    Bundle multiple hypervectors together.

    Args:
        *hypervectors: Hypervectors to bundle

    Returns:
        A new bundled Hypervector
    """
    if not hypervectors:
        raise ValueError("At least one hypervector required")
    return hypervectors[0].bundle(*hypervectors[1:])


def bind(*hypervectors: Hypervector) -> Hypervector:
    """
    Bind multiple hypervectors together.

    Args:
        *hypervectors: Hypervectors to bind

    Returns:
        A new bound Hypervector
    """
    if not hypervectors:
        raise ValueError("At least one hypervector required")
    return hypervectors[0].bind(*hypervectors[1:])


def stack(hypervectors: "list[Hypervector]") -> Hypervector:
    """
    Combine hypervectors/batches into one dimension-first ``(D, N)`` Hypervector.

    Backend-agnostic (numpy or torch). Concatenates along the batch axis (axis 1);
    a 1D ``(D,)`` vector is treated as a single column ``(D, 1)``. For example,
    ``stack([prototype, codebook])`` with a ``(D,)`` prototype and a ``(D, N)``
    codebook returns a ``(D, N + 1)`` Hypervector with the prototype as column 0.

    Args:
        hypervectors: A non-empty list of Hypervectors sharing a backend (and,
            ideally, an encoding).

    Returns:
        A new Hypervector with the inputs concatenated along the batch axis.

    Raises:
        ValueError: If the list is empty or the backends differ.
    """
    if not hypervectors:
        raise ValueError("At least one hypervector required")

    first = hypervectors[0]
    backend = first.backend
    encoding = first.encoding

    arrays = []
    for hv in hypervectors:
        if hv.backend != backend:
            raise ValueError(
                f"Backend mismatch in stack: {backend} vs {hv.backend}. "
                f"Use .to_numpy() or .to_torch() to convert."
            )
        if hv.encoding.__class__ != encoding.__class__:
            warnings.warn(
                f"Encoding mismatch in stack: {encoding.__class__.__name__} vs "
                f"{hv.encoding.__class__.__name__}"
            )
        arrays.append(hv.data)

    if backend == "torch":
        result = torch.column_stack(arrays)
    else:
        result = np.column_stack(arrays)

    return Hypervector(result, encoding, backend)
