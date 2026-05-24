"""Tests for the HDC exception hierarchy."""

import pytest

import pyhdc


class TestExceptionHierarchy:
    def test_all_exceptions_inherit_from_hdc_exception(self):
        assert issubclass(pyhdc.DimensionsNotMatchingError, pyhdc.HDCException)
        assert issubclass(pyhdc.DtypesNotMatchingError, pyhdc.HDCException)
        assert issubclass(pyhdc.GeneratorNotSupportedError, pyhdc.HDCException)
        assert issubclass(pyhdc.RecoveryError, pyhdc.HDCException)
        assert issubclass(pyhdc.RecoveryNotConvergedError, pyhdc.RecoveryError)
        assert issubclass(pyhdc.RecoveryNotSupportedError, pyhdc.RecoveryError)

    def test_hdc_exception_is_exception(self):
        assert issubclass(pyhdc.HDCException, Exception)

    def test_recovery_not_converged_is_recovery_error(self):
        assert issubclass(pyhdc.RecoveryNotConvergedError, pyhdc.RecoveryError)


class TestDimensionsNotMatchingError:
    def test_raise_and_catch(self):
        with pytest.raises(pyhdc.DimensionsNotMatchingError):
            raise pyhdc.DimensionsNotMatchingError(100, 200, "bundle")

    def test_attributes_stored(self):
        try:
            raise pyhdc.DimensionsNotMatchingError(100, 200, "bundle")
        except pyhdc.DimensionsNotMatchingError as e:
            assert e.dim1 == 100
            assert e.dim2 == 200
            assert "bundle" in str(e)

    def test_is_hdc_exception(self):
        with pytest.raises(pyhdc.HDCException):
            raise pyhdc.DimensionsNotMatchingError(1, 2, "test")


class TestDtypesNotMatchingError:
    def test_raise_and_catch(self):
        import numpy as np

        with pytest.raises(pyhdc.DtypesNotMatchingError):
            raise pyhdc.DtypesNotMatchingError(np.float32, np.int32, "bind")

    def test_attributes_stored(self):
        import numpy as np

        try:
            raise pyhdc.DtypesNotMatchingError(np.float32, np.int32, "bind")
        except pyhdc.DtypesNotMatchingError as e:
            assert e.dtype1 == np.float32
            assert e.dtype2 == np.int32


class TestRecoveryNotConvergedError:
    def test_raise_and_catch(self):
        with pytest.raises(pyhdc.RecoveryNotConvergedError):
            raise pyhdc.RecoveryNotConvergedError(50, 100)

    def test_attributes_stored(self):
        try:
            raise pyhdc.RecoveryNotConvergedError(50, 100)
        except pyhdc.RecoveryNotConvergedError as e:
            assert e.iterations == 50
            assert e.max_iterations == 100

    def test_is_recovery_error(self):
        with pytest.raises(pyhdc.RecoveryError):
            raise pyhdc.RecoveryNotConvergedError(1, 10)


class TestGeneratorNotSupportedError:
    def test_raise_and_catch(self):
        with pytest.raises(pyhdc.GeneratorNotSupportedError):
            raise pyhdc.GeneratorNotSupportedError("TestGen does not support bits")

    def test_is_hdc_exception(self):
        with pytest.raises(pyhdc.HDCException):
            raise pyhdc.GeneratorNotSupportedError("msg")


class TestRecoveryNotSupportedError:
    def test_raise_and_catch(self):
        with pytest.raises(pyhdc.RecoveryNotSupportedError):
            raise pyhdc.RecoveryNotSupportedError("Algorithm not supported")

    def test_is_recovery_error(self):
        with pytest.raises(pyhdc.RecoveryError):
            raise pyhdc.RecoveryNotSupportedError("msg")
