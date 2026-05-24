from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional PyTorch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


from pyhdc.exceptions import GeneratorNotSupportedError
from pyhdc.generation.base import DefaultGenerator, HDCGenerator
from pyhdc.hypervector import BackendManager, EncodingSpec, Hypervector
from pyhdc.types import ArrayLike, Backend, Device


def _unpack_operation_result(
    result: Union[ArrayLike, Tuple[ArrayLike, Dict[str, Any]]], operation_fn: Callable
) -> Tuple[ArrayLike, Dict[str, Any]]:
    """
    Unpack operation result, handling both old and new-style returns.

    Adds 'operation' key with function name to metadata dict.

    Args:
        result: Either raw ArrayLike or (ArrayLike, metadata_dict) tuple
        operation_fn: The operation function that produced this result

    Returns:
        Tuple of (data, metadata_dict)
    """
    if isinstance(result, tuple) and len(result) == 2:
        data, metadata = result
        import functools

        if isinstance(operation_fn, functools.partial):
            op_name = operation_fn.func.__name__
        else:
            op_name = operation_fn.__name__

        if isinstance(metadata, dict):
            enhanced_metadata = {"operation": op_name, **metadata}
            return data, enhanced_metadata
        else:
            # Non-dict secondary return (e.g. MatrixMultiplication returns matrices list)
            return data, {"operation": op_name, "aux": metadata}
    return result, {}


class Encoding(ABC):
    """
    Base class for hypervector encoding schemes.

    An encoding defines how hypervectors are generated and how operations
    (similarity, bundling, binding) are performed on them.
    """

    def __init__(
        self,
        dimension: int = 10_000,
        backend: Backend = "numpy",
        device: Optional[Device] = None,
        dtype: Optional[Any] = None,
        mask: Optional[int] = None,
        generator: Optional[HDCGenerator] = None,
    ) -> None:
        """
        Initialize an encoding scheme.

        Args:
            dimension: Number of dimensions in hypervectors
            backend: Backend to use ('numpy' or 'torch')
            device: Device for PyTorch backend
            dtype: Data type override
            mask: Optional mask value
            generator: Optional custom generator (uses default if None)
        """
        self.dimension = dimension
        self.backend = backend
        self.device = device if backend == "torch" else None

        if backend == "torch" and not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch backend requested but PyTorch is not installed. "
                "Install it with: pip install torch"
            )

        # Set up generator
        self._has_custom_generator = generator is not None
        self._generator = generator if generator is not None else DefaultGenerator()

        # Get encoding specification
        spec = self._get_encoding_spec()

        # Override dtype if provided
        if dtype is not None:
            spec.dtype = dtype
        if mask is not None:
            spec.mask = mask

        self._spec = spec
        self._validate_generator()

    def _validate_generator(self) -> None:
        """Validate that generator supports required output type."""
        output_type = self._spec.generator_output_type

        if output_type == "bits" and not self._generator.supports_bits():
            raise GeneratorNotSupportedError(
                f"Generator {self._generator.__class__.__name__} does not support "
                f"bit generation required by {self.__class__.__name__}"
            )
        elif output_type == "words" and not self._generator.supports_words():
            raise GeneratorNotSupportedError(
                f"Generator {self._generator.__class__.__name__} does not support "
                f"word generation required by {self.__class__.__name__}"
            )
        elif output_type == "floats" and not self._generator.supports_floats():
            raise GeneratorNotSupportedError(
                f"Generator {self._generator.__class__.__name__} does not support "
                f"float generation required by {self.__class__.__name__}"
            )

    @abstractmethod
    def _get_encoding_spec(self) -> EncodingSpec:
        """Get the encoding specification for this encoding type."""

    def _generate_with_generator(self, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """
        Generate data using the custom generator.

        Args:
            size: Size specification

        Returns:
            Generated numpy array
        """
        # Determine total number of elements
        if isinstance(size, int):
            total_elements = size
            shape = (size,)
        elif isinstance(size, tuple):
            total_elements = int(np.prod(size))
            shape = size
        else:
            raise ValueError(f"Invalid size specification: {size}")

        output_type = self._spec.generator_output_type

        # Generate based on output type
        if output_type == "bits":
            data = self._generator.generate_bits(total_elements)
        elif output_type == "words":
            # Determine word size from dtype
            dtype_bits = np.dtype(self._spec.dtype).itemsize * 8
            data = self._generator.generate_words(total_elements, dtype_bits)
        elif output_type == "floats":
            # Determine range based on dtype
            if np.issubdtype(self._spec.dtype, np.integer):
                # For integer dtypes, generate in [0, 1] then scale
                floats = self._generator.generate_floats(total_elements, 0.0, 1.0)
                # Convert to appropriate range
                if (
                    self._spec.dtype == np.int8
                    or "bipolar" in self._spec.element_generator.__name__.lower()
                ):
                    # Bipolar: map to {-1, 1}
                    data = [int(2 * round(f) - 1) for f in floats]
                else:
                    # Binary: map to {0, 1}
                    data = [int(round(f)) for f in floats]
            else:
                # For float dtypes, generate in [-1, 1] or appropriate range
                data = self._generator.generate_floats(total_elements, -1.0, 1.0)
        else:
            raise ValueError(f"Unknown output type: {output_type}")

        # Convert to numpy array and reshape
        arr = np.array(data, dtype=self._spec.dtype)
        return arr.reshape(shape)

    def generate(
        self,
        size: Union[int, Tuple[int, ...]] = None,
        backend: Optional[Backend] = None,
        device: Optional[Device] = None,
        use_generator: Optional[bool] = None,
    ) -> Hypervector:
        """
        Generate random hypervector(s).

        Args:
            size: Size specification. If int, generates (size,) shaped array.
                  If tuple, generates that shape. If None, uses self.dimension.
            backend: Backend override
            device: Device override (for PyTorch)
            use_generator: Whether to use the HDCGenerator pathway. Defaults to
                          True if a custom generator was passed at construction,
                          False otherwise (uses element_generator directly, which
                          gives the correct per-encoding distribution).

        Returns:
            A new Hypervector
        """
        if backend is None:
            backend = self.backend
        if device is None:
            device = self.device

        if use_generator is None:
            use_generator = self._has_custom_generator

        # Determine dimensions
        if size is None:
            dimensions = self.dimension
        elif isinstance(size, int):
            dimensions = size
        elif isinstance(size, tuple):
            dimensions = size
        else:
            raise ValueError(f"Invalid size specification: {size}")

        # Generate data
        if use_generator and self._generator is not None:
            data = self._generate_with_generator(dimensions)
        else:
            # Use default element generator
            data = self._spec.element_generator(dimensions, self._spec.dtype)

        # Convert to appropriate backend
        if backend == "torch":
            data = BackendManager.to_torch(data, device)

        return Hypervector(data, self, backend, None)

    def zeros(
        self,
        size: Union[int, Tuple[int, ...]] = None,
        backend: Optional[Backend] = None,
        device: Optional[Device] = None,
    ) -> Hypervector:
        """Generate zero hypervector(s)."""
        if size is None:
            size = self.dimension

        if backend is None:
            backend = self.backend
        if device is None:
            device = self.device

        if backend == "torch":
            data = torch.zeros(size, dtype=self._spec.dtype, device=device)
        else:
            data = np.zeros(size, dtype=self._spec.dtype)

        return Hypervector(data, self, backend, None)

    def from_array(
        self, array: ArrayLike, backend: Optional[Backend] = None
    ) -> Hypervector:
        """Create a Hypervector from an existing array."""
        if backend is None:
            backend = BackendManager.get_backend(array)
        return Hypervector(array, self, backend, None)

    def set_generator(self, generator: HDCGenerator) -> None:
        """
        Set a new generator for this encoding.

        Args:
            generator: The new generator to use

        Raises:
            GeneratorNotSupportedError: If generator doesn't support required output type
        """
        self._generator = generator
        self._validate_generator()

    def get_generator(self) -> HDCGenerator:
        """Get the current generator."""
        return self._generator

    def similarity(
        self,
        hvA: Union[ArrayLike, Hypervector, List],
        hvB: Union[ArrayLike, Hypervector, List],
    ) -> Union[float, ArrayLike, List[Union[float, ArrayLike]]]:
        """
        Compute similarity between hypervector(s).

        Supports batching: if both inputs are lists of equal length, computes
        pairwise similarities element-by-element.

        Args:
            hvA: First hypervector(s) (Hypervector, array, or list)
            hvB: Second hypervector(s) (Hypervector, array, or list)

        Returns:
            Single similarity score (if single inputs) or List of scores (if list inputs)

        Examples:
            >>> # Single similarity
            >>> bsc.similarity(hv1, hv2)  # Returns: float

            >>> # Batched pairwise similarity
            >>> bsc.similarity([hv1, hv2, hv3], [hv4, hv5, hv6])
            >>> # Returns: [sim(hv1,hv4), sim(hv2,hv5), sim(hv3,hv6)]
        """
        from pyhdc.components.input_formatting import _extract_data

        # Batched if both are lists of equal length
        if isinstance(hvA, list) and isinstance(hvB, list):
            if len(hvA) != len(hvB):
                raise ValueError(
                    f"Batched similarity requires equal-length lists. "
                    f"Got {len(hvA)} and {len(hvB)}"
                )

            results = []
            for a, b in zip(hvA, hvB):
                data_a = _extract_data(a)
                data_b = _extract_data(b)
                sim = self._spec.similarity_fn(data_a, data_b)
                results.append(sim)
            return results
        else:
            # Single operation
            data_a = _extract_data(hvA)
            data_b = _extract_data(hvB)
            return self._spec.similarity_fn(data_a, data_b)

    def bundle(
        self,
        *hypervectors: Union[ArrayLike, Hypervector, List],
        batch_dim: Optional[int] = None,
    ) -> Union[Hypervector, List[Hypervector]]:
        """
        Bundle multiple hypervectors, optionally in batches.

        Args:
            *hypervectors: Hypervector objects, raw arrays, or lists to bundle
            batch_dim: If provided with 3D+ array, split along this dimension for batching

        Returns:
            Single Hypervector (if not batched) or List of Hypervectors (if batched)

        Examples:
            >>> # Single bundle (current behavior)
            >>> bsc.bundle(hv1, hv2, hv3)  # Returns: Hypervector
            >>> bsc.bundle([hv1, hv2, hv3])  # Returns: Hypervector

            >>> # Batched bundles (new)
            >>> bsc.bundle([[hv1, hv2], [hv3, hv4]])  # Returns: [bundled1, bundled2]
            >>> bsc.bundle(array_3d, batch_dim=0)  # Returns: list of bundled hypervectors
        """
        from pyhdc.components.input_formatting import (
            _detect_batch_structure,
            _normalize_inputs,
        )

        # Detect if this is a batched operation
        is_batched, groups = _detect_batch_structure(*hypervectors, batch_dim=batch_dim)

        if is_batched:
            # Process each batch group independently
            results = []
            for group in groups:
                # Normalize each group
                if isinstance(group, (list, tuple)):
                    data_arrays, _, _ = _normalize_inputs(*group)
                else:
                    data_arrays, _, _ = _normalize_inputs(group)

                result = self._spec.bundling_fn(*data_arrays)
                result_data, metadata = _unpack_operation_result(
                    result, self._spec.bundling_fn
                )
                results.append(Hypervector(result_data, self, self.backend, metadata))
            return results
        else:
            # Single operation
            if isinstance(groups, (list, tuple)) and len(groups) > 0:
                data_arrays, _, _ = _normalize_inputs(*groups)
            else:
                data_arrays, _, _ = _normalize_inputs(groups)

            result = self._spec.bundling_fn(*data_arrays)
            result_data, metadata = _unpack_operation_result(
                result, self._spec.bundling_fn
            )
            return Hypervector(result_data, self, self.backend, metadata)

    def thin(
        self, hypervector: Union[ArrayLike, Hypervector, List]
    ) -> Union[Hypervector, List[Hypervector]]:
        """
        Apply thinning to hypervector(s).

        Supports batching: if a list is provided, applies thinning independently
        to each hypervector in the list.

        Args:
            hypervector: Hypervector object, raw array, or list of hypervectors to thin

        Returns:
            Single Hypervector (if single input) or List of Hypervectors (if list input)

        Examples:
            >>> # Single thinning
            >>> bsc.thin(hv)  # Returns: Hypervector

            >>> # Batched thinning
            >>> bsc.thin([hv1, hv2, hv3])  # Returns: [thinned1, thinned2, thinned3]
        """
        from pyhdc.components.input_formatting import _extract_data

        # Batched if list input
        if isinstance(hypervector, list):
            results = []
            for hv in hypervector:
                data = _extract_data(hv)
                result = self._spec.thinning_fn(data)
                result_data, metadata = _unpack_operation_result(
                    result, self._spec.thinning_fn
                )
                results.append(Hypervector(result_data, self, self.backend, metadata))
            return results
        else:
            # Single operation
            data = _extract_data(hypervector)
            result = self._spec.thinning_fn(data)
            result_data, metadata = _unpack_operation_result(
                result, self._spec.thinning_fn
            )
            return Hypervector(result_data, self, self.backend, metadata)

    def bind(
        self,
        *hypervectors: Union[ArrayLike, Hypervector, List],
        batch_dim: Optional[int] = None,
    ) -> Union[Hypervector, List[Hypervector]]:
        """
        Bind multiple hypervectors, optionally in batches.

        Args:
            *hypervectors: Hypervector objects, raw arrays, or lists to bind
            batch_dim: If provided with 3D+ array, split along this dimension for batching

        Returns:
            Single Hypervector (if not batched) or List of Hypervectors (if batched)
        """
        from pyhdc.components.input_formatting import (
            _detect_batch_structure,
            _normalize_inputs,
        )

        # Detect if this is a batched operation
        is_batched, groups = _detect_batch_structure(*hypervectors, batch_dim=batch_dim)

        if is_batched:
            # Process each batch group independently
            results = []
            for group in groups:
                # Normalize each group
                if isinstance(group, (list, tuple)):
                    data_arrays, _, _ = _normalize_inputs(*group)
                else:
                    data_arrays, _, _ = _normalize_inputs(group)

                result = self._spec.binding_fn(*data_arrays)
                result_data, metadata = _unpack_operation_result(
                    result, self._spec.binding_fn
                )
                results.append(Hypervector(result_data, self, self.backend, metadata))
            return results
        else:
            # Single operation
            if isinstance(groups, (list, tuple)) and len(groups) > 0:
                data_arrays, _, _ = _normalize_inputs(*groups)
            else:
                data_arrays, _, _ = _normalize_inputs(groups)

            result = self._spec.binding_fn(*data_arrays)
            result_data, metadata = _unpack_operation_result(
                result, self._spec.binding_fn
            )
            return Hypervector(result_data, self, self.backend, metadata)

    def unbind(
        self,
        *hypervectors: Union[ArrayLike, Hypervector, List],
        batch_dim: Optional[int] = None,
    ) -> Union[Hypervector, List[Hypervector]]:
        """
        Unbind hypervectors, optionally in batches.

        Args:
            *hypervectors: Hypervector objects, raw arrays, or lists to unbind
            batch_dim: If provided with 3D+ array, split along this dimension for batching

        Returns:
            Single Hypervector (if not batched) or List of Hypervectors (if batched)
        """
        from pyhdc.components.input_formatting import (
            _detect_batch_structure,
            _normalize_inputs,
        )

        # Detect if this is a batched operation
        is_batched, groups = _detect_batch_structure(*hypervectors, batch_dim=batch_dim)

        if is_batched:
            # Process each batch group independently
            results = []
            for group in groups:
                # Normalize each group
                if isinstance(group, (list, tuple)):
                    data_arrays, _, _ = _normalize_inputs(*group)
                else:
                    data_arrays, _, _ = _normalize_inputs(group)

                result = self._spec.unbinding_fn(*data_arrays)
                result_data, metadata = _unpack_operation_result(
                    result, self._spec.unbinding_fn
                )
                results.append(Hypervector(result_data, self, self.backend, metadata))
            return results
        else:
            # Single operation
            if isinstance(groups, (list, tuple)) and len(groups) > 0:
                data_arrays, _, _ = _normalize_inputs(*groups)
            else:
                data_arrays, _, _ = _normalize_inputs(groups)

            result = self._spec.unbinding_fn(*data_arrays)
            result_data, metadata = _unpack_operation_result(
                result, self._spec.unbinding_fn
            )
            return Hypervector(result_data, self, self.backend, metadata)
