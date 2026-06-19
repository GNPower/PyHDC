#!/usr/bin/env python
"""Smoke tests for top-level pyhdc public API."""

import pyhdc


def test_package_version():
    assert hasattr(pyhdc, "__version__")
    assert isinstance(pyhdc.__version__, str)
    assert len(pyhdc.__version__) > 0


def test_encoding_classes_importable():
    names = [
        "MAP_C",
        "MAP_I",
        "MAP_I_Bits",
        "MAP_B",
        "HRR",
        "HRR_NoNorm",
        "HRR_ConstNorm",
        "FHRR",
        "VTB",
        "MBAT",
        "BSC",
        "BSDC_CDT",
        "BSDC_S",
        "BSDC_SEG",
        "BSDC_THIN",
    ]
    for name in names:
        assert hasattr(pyhdc, name), f"Missing encoding: {name}"


def test_exception_classes_importable():
    names = [
        "HDCException",
        "DimensionsNotMatchingError",
        "DtypesNotMatchingError",
        "GeneratorNotSupportedError",
        "RecoveryError",
        "RecoveryNotConvergedError",
        "RecoveryNotSupportedError",
    ]
    for name in names:
        assert hasattr(pyhdc, name), f"Missing exception: {name}"


def test_convenience_functions_importable():
    names = [
        "generate",
        "zeros",
        "bundle",
        "bind",
        "unbind",
        "stack",
        "permute",
        "inverse",
        "negative",
        "normalize",
    ]
    for name in names:
        assert hasattr(pyhdc, name), f"Missing function: {name}"


def test_core_classes_importable():
    assert hasattr(pyhdc, "Hypervector")
    assert hasattr(pyhdc, "BackendManager")
    assert hasattr(pyhdc, "EncodingSpec")


def test_type_aliases_importable():
    assert hasattr(pyhdc, "ArrayLike")
    assert hasattr(pyhdc, "Backend")
    assert hasattr(pyhdc, "Device")


def test_submodules_importable():
    import pyhdc.components  # noqa: F401
    import pyhdc.generation  # noqa: F401
    import pyhdc.recovery  # noqa: F401


def test_torch_available_flag_exists():
    assert hasattr(pyhdc, "TORCH_AVAILABLE")
    assert isinstance(pyhdc.TORCH_AVAILABLE, bool)
