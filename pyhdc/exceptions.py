#!/usr/bin/env python
"""
HDC Library Exceptions

Custom exceptions for the hyperdimensional computing library.
"""


class HDCException(Exception):
    """Base exception for HDC library."""


class DimensionsNotMatchingError(HDCException):
    """Raised when hypervector dimensions don't match for an operation."""

    def __init__(self, dim1: int, dim2: int, operation: str = "operation"):
        self.dim1 = dim1
        self.dim2 = dim2
        self.operation = operation
        super().__init__(f"Dimension mismatch in {operation}: {dim1} != {dim2}")


class DtypesNotMatchingError(HDCException):
    """Raised when hypervector data types don't match for an operation."""

    def __init__(self, dtype1, dtype2, operation: str = "operation"):
        self.dtype1 = dtype1
        self.dtype2 = dtype2
        self.operation = operation
        super().__init__(f"Data type mismatch in {operation}: {dtype1} != {dtype2}")


class GeneratorNotSupportedError(HDCException):
    """Raised when a generator doesn't support required operations."""


class RecoveryError(HDCException):
    """Base exception for recovery-related errors."""


class RecoveryNotConvergedError(RecoveryError):
    """Raised when recovery algorithm doesn't converge."""

    def __init__(self, iterations: int, max_iterations: int):
        self.iterations = iterations
        self.max_iterations = max_iterations
        super().__init__(
            f"Recovery did not converge after {iterations} iterations "
            f"(max: {max_iterations})"
        )


class RecoveryNotSupportedError(RecoveryError):
    """Raised when recovery is not supported for an encoding."""


def RaiseNotImplementedError(*args, **kwargs):
    """
    Helper function to raise NotImplementedError.
    Used as a placeholder for operations that are not implemented.
    """
    raise NotImplementedError(
        "This operation is not implemented for this encoding scheme."
    )
