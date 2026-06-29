"""Regression tests for pyhdc.components.input_formatting helpers."""

import numpy as np

import pyhdc
from pyhdc.components.input_formatting import _is_hypervector


class _RaisesOnData:  # pylint: disable=too-few-public-methods
    """A non-hypervector whose ``.data`` access raises (not ``AttributeError``).

    Mirrors a raw array that has no buffer protocol: numpy raises
    ``ValueError: cannot include dtype 'E' in a buffer`` when an ml_dtypes array
    (e.g. bfloat16, char kind ``'E'``) exposes ``.data``. ``hasattr`` only
    catches ``AttributeError``, so checking ``.data`` first would let that
    ValueError propagate and crash bundling. The object has no
    ``encoding``/``backend``, so ``_is_hypervector`` must return False before
    ever touching ``.data``.
    """

    @property
    def data(self):
        """Raise on access, mimicking numpy's buffer error for ml_dtypes."""
        raise ValueError("cannot include dtype 'E' in a buffer")


def test_is_hypervector_short_circuits_before_touching_data():
    # Must return False without evaluating .data (which raises ValueError),
    # because the object has no encoding/backend.
    assert _is_hypervector(_RaisesOnData()) is False


def test_is_hypervector_false_for_raw_ndarray():
    assert _is_hypervector(np.zeros(8)) is False


def test_is_hypervector_true_for_hypervector():
    hv = pyhdc.MAP_C(dimension=64).generate()
    assert _is_hypervector(hv) is True
