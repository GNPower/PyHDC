from math import ceil

import numpy as np


def Sumset(hypervector: np.ndarray, density: float = 0.5) -> np.ndarray:
    """Sumset Sets largest values of hypervector to 1 subject to density constraints

    Sets the n largest elements in the hypervector to 1, all other elements
    to 0. n is selected as len(hypervector) * density. If multiple elememts
    are the same value, the lower indices in the hypervector will be selected
    first.

    :param hypervector: The hypervector to thin
    :type hypervector: np.ndarray
    :param density: Required density of the output hypervector, defaults to 0.5
    :type density: float, optional
    :return: Thinned hypervector
    :rtype: np.ndarray
    """
    num_nonzero = ceil(hypervector.size * density)
    # Gets the indices of the largest values in the array
    indices = np.argpartition(hypervector, -num_nonzero)[-num_nonzero:]
    hypervector.fill(0)
    # Sets all selected indices of the hypervector to 1
    hypervector[indices] = 1
    return hypervector


def SegmentedSumset(hypervector: np.ndarray, density: float = 0.5) -> np.ndarray:
    # TODO
    pass
