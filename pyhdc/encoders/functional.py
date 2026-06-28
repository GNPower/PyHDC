#!/usr/bin/env python
"""Functional encoders: a transform of a feature vector.

Each holds a ``(D, F)`` or ``(D,)`` parameter array and maps a feature vector ``(F,)``
(or a batch ``(F, B)``) to a ``(D, B)`` hypervector, dimension-first.

- ``Projection`` is a random linear projection mapped into the target domain via the
  encoding's ``normalize_fn`` (sign for MAP, L2 for HRR/VTB/MBAT, wrap for FHRR).
- ``Sinusoid`` is a random-Fourier-feature map. Real-valued, pairs with HRR/FHRR.
- ``Density`` is a population (threshold) code. Supported for discrete families only.
- ``FractionalPower`` raises the base atom to a fractional power. Supported for
  FHRR (phase scaling) and the HRR family (FFT) only.
"""

import numpy as np

from pyhdc.components.basis.domain import family_endpoints
from pyhdc.encoders import base
from pyhdc.encodings.holographic import FHRR, HRR, HRR_ConstNorm, HRR_NoNorm
from pyhdc.exceptions import RaiseNotImplementedError

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None


class Projection(base.Encoder):
    """Random linear projection of a feature vector, mapped into the target domain.

    ``encode(x) = normalize_fn(W @ x)`` with ``W`` a ``(D, F)`` Gaussian matrix.
    Supported for families with a defined ``normalize_fn`` (MAP, HRR, VTB, MBAT,
    FHRR), raises an exception at construction for BSC and the BSDC family
    (no normalize).
    """

    def __init__(self, encoding, features):
        self.features = int(features)
        if encoding._spec.normalize_fn is RaiseNotImplementedError:
            raise NotImplementedError(
                f"Projection requires an encoding with a defined normalize_fn "
                f"(MAP, HRR, VTB, MBAT, or FHRR); {encoding.__class__.__name__} "
                f"has none."
            )
        super().__init__(encoding)

    def _build_params(self):
        d, f = self.dimension, self.features
        if self.backend == "torch":
            assert torch is not None
            return torch.randn(d, f, device=self.device)
        return np.random.normal(0.0, 1.0, size=(d, f)).astype(np.float32)

    def encode(self, value):
        x, was_single = self._feature_batch(value)  # (F, B)
        proj = self._params @ x  # (D, B)
        out = self.encoding._spec.normalize_fn(proj)
        return self._wrap(out[:, 0] if was_single else out)


class Sinusoid(base.Encoder):
    """Random-Fourier-feature encoder: ``cos(W @ x + b) * sqrt(2 / D)``.

    The output is real-valued (a kernel feature map), so it is meant for the
    cosine-similarity families (HRR and the other real-valued encodings), not for
    FHRR's phase domain: a real ``Sinusoid`` vector read as FHRR phases would sit
    near 0 and make every pair look identical. ``W`` is a ``(D, F)`` Gaussian (scaled
    by ``bandwidth``), ``b`` a ``(D,)`` uniform phase in ``[0, 2*pi)``.
    """

    def __init__(self, encoding, features, bandwidth=1.0):
        self.features = int(features)
        self.bandwidth = float(bandwidth)
        super().__init__(encoding)

    def _build_params(self):
        d, f = self.dimension, self.features
        if self.backend == "torch":
            assert torch is not None
            weight = torch.randn(d, f, device=self.device) * self.bandwidth
            bias = torch.rand(d, device=self.device) * (2 * np.pi)
            return weight, bias
        weight = np.random.normal(0.0, self.bandwidth, size=(d, f)).astype(np.float32)
        bias = np.random.uniform(0.0, 2 * np.pi, d).astype(np.float32)
        return weight, bias

    def encode(self, value):
        x, was_single = self._feature_batch(value)  # (F, B)
        weight, bias = self._params
        scale = np.sqrt(2.0 / self.dimension)
        if self.backend == "torch":
            assert torch is not None
            out = torch.cos(weight @ x + bias[:, None]) * scale
        else:
            out = np.cos(weight @ x + bias[:, None]) * scale
        return self._wrap(out[:, 0] if was_single else out)


class Density(base.Encoder):
    """Population (threshold) code over a real value. Discrete families only.

    Holds ``(D,)`` random thresholds in ``[low, high]``, a coordinate takes the
    high endpoint where ``value >= threshold``, else the low endpoint.
    """

    def __init__(self, encoding, low=0.0, high=1.0):
        if high <= low:
            raise ValueError(f"high must be > low, got low={low}, high={high}")
        self.low = float(low)
        self.high = float(high)
        self._lo, self._hi = family_endpoints(encoding)  # raises for continuous
        super().__init__(encoding)

    def _build_params(self):
        d = self.dimension
        if self.backend == "torch":
            assert torch is not None
            return torch.rand(d, device=self.device) * (self.high - self.low) + self.low
        return np.random.uniform(self.low, self.high, d).astype(np.float32)

    def encode(self, value):
        values, was_scalar = self._value_batch(value)  # (B,)
        thresholds = self._params[:, None]  # (D, 1)
        if self.backend == "torch":
            assert torch is not None
            vals = torch.as_tensor(
                np.asarray(values, dtype=np.float32), device=self.device
            )
            fired = vals[None, :] >= thresholds  # (D, B)
            ref = self.encoding.generate(1).data
            out = (fired.to(ref.dtype) * (self._hi - self._lo) + self._lo).to(ref.dtype)
        else:
            fired = np.asarray(values, dtype=np.float32)[None, :] >= thresholds
            out = np.where(fired, self._hi, self._lo).astype(self.encoding._spec.dtype)
        return self._wrap(out[:, 0] if was_scalar else out)


class FractionalPower(base.Encoder):
    """Fractional-power encoding (FPE): the base atom raised to a fractional power.

    FHRR scales the stored phases (``(e^{i*theta})^v = e^{i*v*theta}``), the HRR family
    raises the FFT of the atom to the power. Defined only for FHRR and the HRR family,
    raises ``NotImplementedError`` at construction for every other family.
    """

    _SUPPORTED = (FHRR, HRR, HRR_NoNorm, HRR_ConstNorm)

    def __init__(self, encoding):
        if not isinstance(encoding, self._SUPPORTED):
            raise NotImplementedError(
                f"FractionalPower is defined only for FHRR (phase scaling) and the HRR "
                f"family (FFT), not {encoding.__class__.__name__}."
            )
        super().__init__(encoding)

    def _build_params(self):
        return self.encoding.generate(self.dimension).data  # (D,) base atom

    def encode(self, value):
        values, was_scalar = self._value_batch(value)  # (B,) exponents
        atom = self._params
        if isinstance(self.encoding, FHRR):
            out = self._encode_phase(atom, values)
        else:
            out = self._encode_fft(atom, values)
        return self._wrap(out[:, 0] if was_scalar else out)

    def _encode_phase(self, atom, values):
        if self.backend == "torch":
            assert torch is not None
            exps = torch.as_tensor(
                np.asarray(values, dtype=np.float32), device=self.device
            )
            scaled = atom[:, None] * exps[None, :]
        else:
            scaled = atom[:, None] * np.asarray(values, dtype=np.float32)[None, :]
        return self.encoding._spec.normalize_fn(scaled)  # WrapPhase -> [-pi, pi)

    def _encode_fft(self, atom, values):
        if self.backend == "torch":
            assert torch is not None
            exps = torch.as_tensor(
                np.asarray(values, dtype=np.float32), device=self.device
            )
            spec = torch.fft.fft(atom)
            powered = spec[:, None] ** exps[None, :].to(spec.dtype)
            return torch.fft.ifft(powered, dim=0).real.to(atom.dtype)
        spec = np.fft.fft(np.asarray(atom))
        powered = spec[:, None] ** np.asarray(values, dtype=np.float64)[None, :]
        return np.fft.ifft(powered, axis=0).real.astype(np.float32)
