from abc import ABC, abstractmethod
from math import pi, sqrt
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional PyTorch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


from pyhdc.components.binding import (
    ElementAngleAddition,
    ElementAngleSubtraction,
    ElementMultiplication,
    ExclusiveOr,
)
from pyhdc.components.elements import (
    BernoulliBinary,
    BernoulliBipolar,
    BernoulliSparse,
    NormalReal,
    UniformAngles,
    UniformBipolar,
)
from pyhdc.components.similarity import (
    AngleDistance,
    CosineSimilarity,
    HammingDistance,
    Overlap,
)
from pyhdc.components.unary import CyclicShift
from pyhdc.config import get_default_backend, get_default_device
from pyhdc.exceptions import GeneratorNotSupportedError
from pyhdc.generation.base import DefaultGenerator, HDCGenerator
from pyhdc.hypervector import BackendManager, EncodingSpec, Hypervector
from pyhdc.types import ArrayLike, Backend, Device

# Element generators eligible for the vectorized batched-generation fast path.
# Each is a pure i.i.d. per-element draw, so the whole batch is drawn in one
# (D, *batch) call (see Encoding._fast_path_draw).
_IID_ELEMENT_GENERATORS = frozenset(
    {
        BernoulliBipolar,
        BernoulliBinary,
        UniformBipolar,
        UniformAngles,
        NormalReal,
        BernoulliSparse,
    }
)

# Binders that handle a batched ``(D, *batch)`` input natively (broadcast or
# vectorize). Any other binder is applied per column when bind/unbind receives a
# batch, so a batched call still returns a single batched Hypervector.
_BATCH_SAFE_BINDERS = frozenset(
    {
        ElementMultiplication,
        ExclusiveOr,
        ElementAngleAddition,
        ElementAngleSubtraction,
    }
)

# Similarity functions that implement the matmul-backed cross mode (the (D, P) x
# (D, M) -> (P, M) outer product). An encoding whose similarity_fn is outside this
# set raises NotImplementedError on mode="cross" so the caller can fall back to a
# per-pair oracle.
SUPPORTED_CROSS_SIMILARITIES = frozenset(
    {CosineSimilarity, HammingDistance, Overlap, AngleDistance}
)


def _warn_batch_dim() -> None:
    """Emit the deprecation warning for the ``batch_dim`` parameter."""
    import warnings

    warnings.warn(
        "batch_dim is deprecated and will be removed in a future release. Pass a "
        "batched array directly (bundle/bind/unbind batch automatically) or use "
        "axis= on bundle.",
        DeprecationWarning,
        stacklevel=3,
    )


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
            # Non-dict secondary return
            # (e.g. MatrixMultiplication returns matrices list)
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
        backend: Optional[Backend] = None,
        device: Optional[Device] = None,
        dtype: Optional[Any] = None,
        mask: Optional[int] = None,
        generator: Optional[HDCGenerator] = None,
        similarity_remap: Optional[Callable] = None,
    ) -> None:
        """
        Initialize an encoding scheme.

        Args:
            dimension: Number of dimensions in hypervectors
            backend: Backend to use ('numpy' or 'torch'). Defaults to the global
                preference (see ``pyhdc.prefer_torch``/``prefer_numpy``), which is
                'numpy' unless changed.
            device: Device for PyTorch backend. Defaults to the global preference
                (see ``pyhdc.prefer_cuda``/``prefer_cpu``).
            dtype: Data type override
            mask: Optional mask value
            generator: Optional custom generator (uses default if None)
            similarity_remap: Optional callable applied to every similarity result
                before returning. All similarity functions return [-1, 1] by default;
                use this to remap the output, e.g.
                ``pyhdc.components.similarity.remap_to_unit``
                to shift to [0, 1].
        """
        if backend is None:
            backend = get_default_backend()
        if device is None:
            device = get_default_device()

        self.dimension = dimension
        self.backend = backend
        self.device = device if backend == "torch" else None
        self._similarity_remap = similarity_remap

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

    def _generate_one(self, dim: int, use_generator: bool) -> np.ndarray:
        """
        Generate a single ``(dim,)`` hypervector as a numpy array.

        Args:
            dim: Hypervector dimension.
            use_generator: Whether to use the custom HDCGenerator pathway.

        Returns:
            A 1D numpy array of length ``dim``.
        """
        if use_generator and self._generator is not None:
            return self._generate_with_generator(dim)
        return self._spec.element_generator(dim, self._spec.dtype)

    def _fast_path_draw(
        self, dim: int, batch: Tuple[int, ...], use_generator: bool
    ) -> Optional[np.ndarray]:
        """
        Vectorized i.i.d. draw for a batch, or ``None`` to use the sequential loop.

        Draws the whole ``(D, *batch)`` array in one call, directly in
        dimension-first layout, for the pure i.i.d. element generators. The result
        is reproducible under a fixed seed for a given batch shape, but is not
        value-identical to generating the vectors one at a time (the RNG stream is
        consumed as a single block rather than per column). Ordered/custom
        generators and ``SparseSegmented`` return ``None`` and fall back to the
        sequential loop.
        """
        if use_generator:
            return None
        gen = self._spec.element_generator
        if gen not in _IID_ELEMENT_GENERATORS:
            return None

        # Draw directly in (D, *batch) output layout. Each call mirrors its element
        # generator keyword-for-keyword so the per-element distribution is identical.
        shape = (dim, *batch)
        if gen is BernoulliBipolar:
            arr = np.random.choice([-1, 1], size=shape, p=[0.5, 0.5])
        elif gen is BernoulliBinary:
            arr = np.random.binomial(size=shape, n=1, p=0.5)
        elif gen is UniformBipolar:
            arr = np.random.uniform(-1, 1, shape)
        elif gen is UniformAngles:
            arr = np.random.uniform(-pi, pi, shape)
        elif gen is NormalReal:
            arr = np.random.normal(0, sqrt(1 / dim), size=shape)
        elif gen is BernoulliSparse:
            arr = np.random.binomial(size=shape, n=1, p=1 / sqrt(dim))
        else:  # pragma: no cover
            return None

        return np.ascontiguousarray(arr, dtype=self._spec.dtype)

    def generate(
        self,
        size: Union[int, Tuple[int, ...]] = None,
        backend: Optional[Backend] = None,
        device: Optional[Device] = None,
        use_generator: Optional[bool] = None,
    ) -> Hypervector:
        """
        Generate random hypervector(s).

        Hypervectors are dimension-first. A scalar (or ``None``) ``size`` produces a
        single ``(D,)`` hypervector; a tuple ``(D, N)`` produces a batch of ``N``
        hypervectors of dimension ``D`` stored as columns of a ``(D, N)`` array
        (and likewise ``(D, N, M)`` for higher-rank batches).

        Batched generation is reproducible under a fixed seed for a given batch
        shape. For the i.i.d. element generators the batch is drawn in one
        vectorized call. Ordered and custom generators draw per vector.

        Args:
            size: ``None`` or int for a single ``(D,)`` vector; a tuple
                ``(D, *batch)`` for a batch of ``prod(batch)`` vectors of dimension
                ``D``.
            backend: Backend override (defaults to the encoding's backend).
            device: Device override for the torch backend.
            use_generator: Whether to use the HDCGenerator pathway. Defaults to True
                if a custom generator was passed at construction, False otherwise
                (uses element_generator directly, which gives the correct
                per-encoding distribution).

        Returns:
            A new Hypervector.
        """
        if backend is None:
            backend = self.backend
        if device is None:
            device = self.device
        if use_generator is None:
            use_generator = self._has_custom_generator

        if size is None or isinstance(size, int):
            dim = self.dimension if size is None else size
            data = self._generate_one(dim, use_generator)
        elif isinstance(size, tuple):
            dim, batch = size[0], size[1:]
            if not batch:
                data = self._generate_one(dim, use_generator)
            else:
                data = self._fast_path_draw(dim, batch, use_generator)
                if data is None:
                    count = 1
                    for axis in batch:
                        count *= int(axis)
                    columns = [
                        self._generate_one(dim, use_generator) for _ in range(count)
                    ]
                    data = np.stack(columns, axis=-1).reshape((dim, *batch))
        else:
            raise ValueError(f"Invalid size specification: {size}")

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

        # Build in numpy first so the encoding's numpy dtype carries over. torch.zeros
        # does not accept a numpy dtype, so convert (preserving dtype) for the torch
        # backend.
        data = np.zeros(size, dtype=self._spec.dtype)
        if backend == "torch":
            data = BackendManager.to_torch(data, device)

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
            GeneratorNotSupportedError: If generator doesn't support required
                output type
        """
        self._generator = generator
        self._validate_generator()

    def get_generator(self) -> HDCGenerator:
        """Get the current generator."""
        return self._generator

    def similarity(
        self,
        hvA: Union[ArrayLike, Hypervector, List],
        hvB: Optional[Union[ArrayLike, Hypervector, List]] = None,
        *,
        axis: Optional[int] = None,
        mode: str = "pairwise",
    ) -> Union[float, ArrayLike, List[Union[float, ArrayLike]]]:
        """
        Compute similarity between hypervector(s).

        Hypervectors are dimension-first ``(D, N)``. Calling conventions:

        - ``similarity(a, b)`` with two ``(D,)`` vectors -> a scalar score
        - ``similarity(A, B)`` with two ``(D, N)`` batches -> ``N`` per-column scores
        - ``similarity(v, B)`` with a vector and a ``(D, N)`` batch -> ``N`` scores
        - ``similarity(batch)`` with one ``(D, N)`` batch -> ``N-1`` scores of
          column 0 against each remaining column
        - ``similarity([..], [..])`` with two equal-length lists -> pairwise scores
        - ``similarity(A, B, mode="cross")`` with ``A=(D, P)`` and ``B=(D, M)`` -> the
          full ``(P, M)`` cross-similarity matrix (every column of A against every
          column of B), matmul-backed

        Args:
            hvA: First hypervector(s) (Hypervector, array, or list), or a single
                ``(D, N)`` batch when ``hvB`` is omitted.
            hvB: Optional second hypervector(s).
            axis: Batch axis to reduce for a single ``(D, N, M, ...)`` batch.
            mode: ``"pairwise"`` (default) or ``"cross"`` for the full outer product.

        Returns:
            A scalar, a 1D array of scores, a ``(P, M)`` matrix (cross), or a list of
            scores (for list inputs).

        Examples:
            >>> bsc.similarity(hv1, hv2)                       # scalar
            >>> enc.similarity(codebook)                       # col 0 vs the rest
            >>> bsc.similarity([hv1, hv2], [hv4, hv5])         # [sim(1,4), sim(2,5)]
            >>> enc.similarity(protos, codebook, mode="cross") # (P, M) matrix
        """
        from pyhdc.components.input_formatting import _extract_data

        if mode == "cross":
            if isinstance(hvA, list) or isinstance(hvB, list):
                raise ValueError(
                    'similarity mode="cross" requires two batch operands, not lists'
                )
            if hvB is None:
                raise ValueError(
                    'similarity mode="cross" requires two batches, A=(D, P) and B=(D, M)'
                )
            if axis is not None:
                raise ValueError('similarity mode="cross" does not accept axis=')
            if self._spec.similarity_fn not in SUPPORTED_CROSS_SIMILARITIES:
                raise NotImplementedError(
                    f"cross similarity is not supported for "
                    f"{self._spec.similarity_fn.__name__}; fall back to a per-pair "
                    f"oracle"
                )
            result = self._spec.similarity_fn(
                _extract_data(hvA), _extract_data(hvB), mode="cross"
            )
            if self._similarity_remap is not None:
                result = self._similarity_remap(result)
            return result

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
                sim = self._spec.similarity_fn(data_a, data_b, axis=axis)
                if self._similarity_remap is not None:
                    sim = self._similarity_remap(sim)
                results.append(sim)
            return results

        # Single (D, N) batch (hvB omitted) or a two-operand comparison
        if hvB is None:
            result = self._spec.similarity_fn(_extract_data(hvA), axis=axis)
        else:
            result = self._spec.similarity_fn(
                _extract_data(hvA), _extract_data(hvB), axis=axis
            )
        if self._similarity_remap is not None:
            result = self._similarity_remap(result)
        return result

    def bundle(
        self,
        *hypervectors: Union[ArrayLike, Hypervector, List],
        axis: Union[None, int, Tuple[int, ...]] = None,
        batch_dim: Optional[int] = None,
    ) -> Union[Hypervector, List[Hypervector]]:
        """
        Bundle multiple hypervectors, optionally in batches.

        Args:
            *hypervectors: Hypervector objects, raw arrays, or lists to bundle
            batch_dim: If provided with 3D+ array, split along this dimension
                for batching

        Returns:
            Single Hypervector (if not batched) or List of Hypervectors (if batched)

        Examples:
            >>> # Single bundle (current behavior)
            >>> bsc.bundle(hv1, hv2, hv3)  # Returns: Hypervector
            >>> bsc.bundle([hv1, hv2, hv3])  # Returns: Hypervector

            >>> # Batched bundles (new)
            >>> bsc.bundle([[hv1, hv2], [hv3, hv4]])  # Returns: [bundled1, bundled2]
            >>> bsc.bundle(array_3d, batch_dim=0)
            ... # Returns: list of bundled hypervectors
            >>> enc.bundle(tensor_DNM, axis=2)  # (D, N, M) -> (D, N)
        """
        if axis is not None and batch_dim is not None:
            raise ValueError("bundle accepts either axis= or batch_dim=, not both")
        if batch_dim is not None:
            _warn_batch_dim()

        from pyhdc.components.input_formatting import (
            _detect_batch_structure,
            _extract_data,
            _normalize_inputs,
        )

        # Vectorized fast path: a single 3D array split by batch_dim is the same
        # as reducing the *other* batch axis in one op, then splitting the result.
        # Taken before _detect_batch_structure so we skip its per-slice split and
        # the per-group loop. Ragged nested lists, batch_dim=0, and 4D+ fall
        # through; the wrapper loop in _split_batch_result does no numerical work.
        if batch_dim in (1, 2) and len(hypervectors) == 1:
            arr = _extract_data(hypervectors[0])
            if getattr(arr, "ndim", 0) == 3:
                reduce_axis = 2 if batch_dim == 1 else 1
                result = self._spec.bundling_fn(arr, axis=reduce_axis)
                result_data, metadata = _unpack_operation_result(
                    result, self._spec.bundling_fn
                )
                return self._split_batch_result(result_data, metadata)

        # Detect if this is a batched operation
        is_batched, groups = _detect_batch_structure(*hypervectors, batch_dim=batch_dim)

        if is_batched:
            # General path: per-group loop (ragged nested lists, other ranks).
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

            result = self._spec.bundling_fn(*data_arrays, axis=axis)
            result_data, metadata = _unpack_operation_result(
                result, self._spec.bundling_fn
            )
            return Hypervector(result_data, self, self.backend, metadata)

    def _split_batch_result(
        self, result_data: ArrayLike, metadata: Dict[str, Any]
    ) -> List[Hypervector]:
        """
        Split a vectorized ``(D, n)`` bundle result into ``n`` ``(D,)`` Hypervectors.

        Backs the ``batch_dim`` fast path. Per-output metadata arrays (whose last
        axis matches the kept batch axis) are sliced per result, scalars and the
        ``operation`` name are shared.
        """
        n = result_data.shape[1]
        results = []
        for j in range(n):
            meta_j = {
                key: (
                    val[..., j]
                    if hasattr(val, "shape")
                    and getattr(val, "ndim", 0) >= 1
                    and val.shape[-1] == n
                    else val
                )
                for key, val in metadata.items()
            }
            results.append(Hypervector(result_data[:, j], self, self.backend, meta_j))
        return results

    def _bind_single(self, fn: Callable, data_arrays: List[ArrayLike]) -> Hypervector:
        """
        Apply a binding function to one set of operands, batching automatically.

        Element-wise binders broadcast a batch natively. Any other binder is
        applied per column over the trailing batch axes, so a batched call still
        returns one batched Hypervector without requiring ``batch_dim``.
        """
        import functools

        base_fn = fn.func if isinstance(fn, functools.partial) else fn
        batched = any(getattr(a, "ndim", 1) > 1 for a in data_arrays)
        if batched and base_fn not in _BATCH_SAFE_BINDERS:
            return self._bind_per_column(fn, data_arrays)
        result = fn(*data_arrays)
        result_data, metadata = _unpack_operation_result(result, fn)
        return Hypervector(result_data, self, self.backend, metadata)

    def _bind_per_column(
        self, fn: Callable, data_arrays: List[ArrayLike]
    ) -> Hypervector:
        """Apply a non-batch-safe binder per column over the trailing batch axes."""
        from pyhdc.components.input_formatting import _broadcast_operands

        is_torch = self.backend == "torch"
        operands = _broadcast_operands(data_arrays, is_torch)
        ref = operands[0]
        dim = ref.shape[0]
        batch_shape = tuple(int(s) for s in ref.shape[1:])
        count = 1
        for size in batch_shape:
            count *= size
        flats = [op.reshape(op.shape[0], -1) for op in operands]

        cols = []
        agg_meta: Dict[str, Any] = {}
        for i in range(count):
            result = fn(*(flat[:, i] for flat in flats))
            res_data, meta = _unpack_operation_result(result, fn)
            cols.append(res_data)
            for key, val in meta.items():
                if key == "operation":
                    agg_meta[key] = val
                else:
                    agg_meta.setdefault(key, []).append(val)

        if is_torch:
            stacked = torch.stack(cols, dim=-1)
        else:
            stacked = np.stack(cols, axis=-1)
        stacked = stacked.reshape((dim, *batch_shape))
        return Hypervector(stacked, self, self.backend, agg_meta)

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

    def permute(
        self, hypervector: Union[ArrayLike, Hypervector], shift: int = 1
    ) -> Hypervector:
        """
        Permute (cyclic-shift) a hypervector along the dimension axis (axis 0).

        Args:
            hypervector: Hypervector or raw array to permute.
            shift: Positions to roll along axis 0; negative inverts the permute.

        Returns:
            A new permuted Hypervector.
        """
        from pyhdc.components.input_formatting import _extract_data

        fn = self._spec.permute_fn or CyclicShift
        data = _extract_data(hypervector)
        result = fn(data, shift=shift)
        result_data, metadata = _unpack_operation_result(result, fn)
        return Hypervector(result_data, self, self.backend, metadata)

    def inverse(self, hypervector: Union[ArrayLike, Hypervector]) -> Hypervector:
        """
        Binding inverse of a hypervector.

        Raises ``NotImplementedError`` for encodings whose binding has no defined
        inverse (e.g. MAP_C continuous, VTB, MBAT, BSDC_*).
        """
        from pyhdc.components.input_formatting import _extract_data

        data = _extract_data(hypervector)
        result = self._spec.inverse_fn(data)
        result_data, metadata = _unpack_operation_result(result, self._spec.inverse_fn)
        return Hypervector(result_data, self, self.backend, metadata)

    def negative(self, hypervector: Union[ArrayLike, Hypervector]) -> Hypervector:
        """
        Bundling (additive) inverse of a hypervector.

        Raises ``NotImplementedError`` for encodings with no defined negative
        (e.g. FHRR, BSC, BSDC_*).
        """
        from pyhdc.components.input_formatting import _extract_data

        data = _extract_data(hypervector)
        result = self._spec.negative_fn(data)
        result_data, metadata = _unpack_operation_result(result, self._spec.negative_fn)
        return Hypervector(result_data, self, self.backend, metadata)

    def normalize(self, hypervector: Union[ArrayLike, Hypervector]) -> Hypervector:
        """
        Normalize a hypervector to its encoding's canonical form (L2 unit length
        for real encodings, bipolar sign for MAP, phase wrap for FHRR).

        Raises ``NotImplementedError`` for encodings with no defined normalization
        (e.g. BSC, BSDC_*).
        """
        from pyhdc.components.input_formatting import _extract_data

        data = _extract_data(hypervector)
        result = self._spec.normalize_fn(data)
        result_data, metadata = _unpack_operation_result(
            result, self._spec.normalize_fn
        )
        return Hypervector(result_data, self, self.backend, metadata)

    def bind(
        self,
        *hypervectors: Union[ArrayLike, Hypervector, List],
        batch_dim: Optional[int] = None,
    ) -> Union[Hypervector, List[Hypervector]]:
        """
        Bind hypervectors, batching automatically when the input is batched.

        A single ``(D, N)`` (or higher-rank) batch is bound in one call: the
        element-wise binders (MAP multiply, BSC xor, FHRR angle add) must broadcast
        natively, and the others (convolution, shifting, matrix, VTB, CDT) are
        applied per column internally, so the result is one batched Hypervector.
        ``batch_dim`` is no longer required.

        Args:
            *hypervectors: Hypervector objects, raw arrays, or lists to bind.
            batch_dim: Deprecated. Splits a 3D+ array along this axis and returns a
                list of results, pass a batched array directly instead.

        Returns:
            A single Hypervector, or a list of Hypervectors for the deprecated
            ``batch_dim`` / nested-list forms.
        """
        if batch_dim is not None:
            _warn_batch_dim()

        from pyhdc.components.input_formatting import (
            _detect_batch_structure,
            _normalize_inputs,
        )

        is_batched, groups = _detect_batch_structure(*hypervectors, batch_dim=batch_dim)
        if is_batched:
            results = []
            for group in groups:
                if isinstance(group, (list, tuple)):
                    data_arrays, _, _ = _normalize_inputs(*group)
                else:
                    data_arrays, _, _ = _normalize_inputs(group)
                results.append(self._bind_single(self._spec.binding_fn, data_arrays))
            return results

        if isinstance(groups, (list, tuple)) and len(groups) > 0:
            data_arrays, _, _ = _normalize_inputs(*groups)
        else:
            data_arrays, _, _ = _normalize_inputs(groups)
        return self._bind_single(self._spec.binding_fn, data_arrays)

    def unbind(
        self,
        *hypervectors: Union[ArrayLike, Hypervector, List],
        batch_dim: Optional[int] = None,
    ) -> Union[Hypervector, List[Hypervector]]:
        """
        Unbind hypervectors, batching automatically when the input is batched.

        Mirrors :meth:`bind`: a single ``(D, N)`` batch is unbound in one call
        (element-wise unbinders broadcast; the others are applied per column), so
        ``batch_dim`` is no longer required.

        Args:
            *hypervectors: Hypervector objects, raw arrays, or lists to unbind.
            batch_dim: Deprecated. Splits a 3D+ array along this axis and returns a
                list of results, pass a batched array directly instead.

        Returns:
            A single Hypervector, or a list of Hypervectors for the deprecated
            ``batch_dim`` / nested-list forms.

        Raises:
            NotImplementedError: For encodings that do not support unbinding.
        """
        if batch_dim is not None:
            _warn_batch_dim()

        from pyhdc.components.input_formatting import (
            _detect_batch_structure,
            _normalize_inputs,
        )

        is_batched, groups = _detect_batch_structure(*hypervectors, batch_dim=batch_dim)
        if is_batched:
            results = []
            for group in groups:
                if isinstance(group, (list, tuple)):
                    data_arrays, _, _ = _normalize_inputs(*group)
                else:
                    data_arrays, _, _ = _normalize_inputs(group)
                results.append(self._bind_single(self._spec.unbinding_fn, data_arrays))
            return results

        if isinstance(groups, (list, tuple)) and len(groups) > 0:
            data_arrays, _, _ = _normalize_inputs(*groups)
        else:
            data_arrays, _, _ = _normalize_inputs(groups)
        return self._bind_single(self._spec.unbinding_fn, data_arrays)
