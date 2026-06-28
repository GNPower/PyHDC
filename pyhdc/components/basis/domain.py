#!/usr/bin/env python
"""Per-family element-domain lookup for basis builders.

``thermometer`` and density-style codes need the two extreme element values of an
encoding's value domain (the "low" and "high" endpoints). These are well defined only
for the discrete families (bipolar and binary). Continuous and phase families have no
natural pair of endpoints and raise ``NotImplementedError``.
"""

from pyhdc.components.binding import (
    CircularConvolution,
    ElementAngleAddition,
    ElementMultiplication,
    ExclusiveOr,
)
from pyhdc.components.elements import (
    BernoulliBinary,
    BernoulliBipolar,
    BernoulliSparse,
    SparseSegmented,
)

# Element-generator object -> (low, high) endpoints of that family's value domain.
# Keyed on the generator object (hashable by identity, every generator is a bare
# module-level function).
#
# Intentionally absent (these fall through to NotImplementedError):
#   UniformBipolar -> MAP_C            (real-continuous)
#   UniformAngles  -> FHRR             (complex/phase)
#   NormalReal     -> HRR, HRR_NoNorm, HRR_ConstNorm, VTB, MBAT (real-continuous)
_ENDPOINTS = {
    BernoulliBipolar: (-1, 1),  # MAP_I, MAP_I_Bits, MAP_B
    BernoulliBinary: (0, 1),  # BSC
    BernoulliSparse: (0, 1),  # BSDC_CDT, BSDC_S, BSDC_THIN
    SparseSegmented: (0, 1),  # BSDC_SEG
}


def family_endpoints(encoding):
    """Return the ``(low, high)`` element endpoints for ``encoding``'s value domain.

    Args:
        encoding: An :class:`~pyhdc.Encoding` instance.

    Returns:
        A ``(low, high)`` tuple of scalars in the family's domain.

    Raises:
        NotImplementedError: If the family is continuous or phase-valued and has no
            discrete endpoint pair (MAP_C, HRR family, VTB, MBAT, FHRR).
    """
    gen = encoding._spec.element_generator
    try:
        return _ENDPOINTS[gen]
    except KeyError:
        raise NotImplementedError(
            f"{encoding.__class__.__name__} (generator {gen.__name__}) has no discrete "
            f"(low, high) endpoint pair, thermometer/density basis require a discrete "
            f"family (bipolar or binary). Continuous and phase families "
            f"(MAP_C, HRR, HRR_NoNorm, HRR_ConstNorm, VTB, MBAT, FHRR) are unsupported."
        ) from None


def binding_identity(encoding, dimension=None):
    """Return the binding-identity element ``e`` (where ``bind(x, e) == x``) as ``(D,)``.

    The neutral element of the encoding's binding rule, in the encoding's value
    domain and backend:

    - element-wise multiply (MAP) -> all ones
    - XOR (BSC) -> all zeros
    - circular convolution (HRR family) -> the impulse ``[1, 0, ..., 0]``
    - angle addition (FHRR) -> zero phase (all zeros)

    Args:
        encoding: An :class:`~pyhdc.Encoding` instance.
        dimension: Hypervector dimension (defaults to ``encoding.dimension``).

    Returns:
        A ``(D,)`` array (numpy or torch) holding the binding-identity element.

    Raises:
        NotImplementedError: For binding rules with no neutral element (VTB, MBAT,
            and the BSDC family).
    """
    dim = encoding.dimension if dimension is None else int(dimension)
    fn = encoding._spec.binding_fn
    base = encoding.zeros(dim).data  # (D,) zeros in the right backend/dtype/device
    if fn is ExclusiveOr or fn is ElementAngleAddition:
        return base  # XOR identity (0) / zero phase
    if fn is ElementMultiplication:
        return base + 1  # multiplicative identity (1)
    if fn is CircularConvolution:
        base[0] = 1  # convolution identity (impulse)
        return base
    raise NotImplementedError(
        f"{encoding.__class__.__name__} (binding {fn.__name__}) has no binding-identity "
        f"element. Identity is defined for the MAP, HRR, FHRR, and BSC families only "
        f"(VTB, MBAT, and the BSDC family have no neutral binding element)."
    )
