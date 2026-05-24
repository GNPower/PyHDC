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
# Convolution-based Binding (HRR)
# ============================================================================


def CircularConvolution(*hypervectors: ArrayLike) -> ArrayLike:
    """
    Circular convolution binding.

    Binds hypervectors using circular convolution in the frequency domain.
    Used in Holographic Reduced Representations (HRR).

    Args:
        *hypervectors: Variable number of hypervectors to bind, or single 2D batch

    Returns:
        Bound hypervector

    Note:
        Binding is performed iteratively for more than 2 vectors:
        bind(A, B, C) = bind(bind(A, B), C)
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)

    if is_torch:
        # PyTorch FFT
        result = torch.fft.ifft(torch.fft.fft(hvs[0]) * torch.fft.fft(hvs[1])).real
        for i in range(2, len(hvs)):
            result = torch.fft.ifft(torch.fft.fft(result) * torch.fft.fft(hvs[i])).real
    else:
        # NumPy FFT
        result = np.real(np.fft.ifft(np.fft.fft(hvs[0]) * np.fft.fft(hvs[1])))
        for i in range(2, len(hvs)):
            result = np.real(np.fft.ifft(np.fft.fft(result) * np.fft.fft(hvs[i])))

    return result


def CircularCorrelation(*hypervectors: ArrayLike) -> ArrayLike:
    """
    Circular correlation (unbinding operation for circular convolution).

    Unbinds hypervectors by performing circular correlation, which is the
    approximate inverse of circular convolution.

    Args:
        *hypervectors: Variable number of hypervectors, or single 2D batch

    Returns:
        Unbound hypervector

    Note:
        For unbinding bind(A, B) with B, compute: correlate(bind(A, B), B) â‰ˆ A
    """
    hvs, is_torch, _ = _normalize_inputs(*hypervectors)

    if is_torch:
        # PyTorch FFT with conjugate
        result = torch.fft.ifft(
            torch.fft.fft(hvs[0]) * torch.conj(torch.fft.fft(hvs[1]))
        ).real
        for i in range(2, len(hvs)):
            result = torch.fft.ifft(
                torch.fft.fft(result) * torch.conj(torch.fft.fft(hvs[i]))
            ).real
    else:
        # NumPy FFT with conjugate
        result = np.real(np.fft.ifft(np.fft.fft(hvs[0]) * np.conj(np.fft.fft(hvs[1]))))
        for i in range(2, len(hvs)):
            result = np.real(
                np.fft.ifft(np.fft.fft(result) * np.conj(np.fft.fft(hvs[i])))
            )

    return result
