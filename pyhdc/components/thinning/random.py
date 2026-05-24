from math import ceil

import numpy as np


def Random(hypervector: np.ndarray, density: float = 0.5) -> np.ndarray:
    """Random Keeps randomly selected nonzero elements of hypervector

    Sets randomly selected nonzero elements of a hypervector
    to 1 subject to a given hypervector density. All other
    elements are set to zero. If the hypervector already
    meets the density constraint the identical hypervector
    is returned.

    :param hypervector: The hypervector to thin
    :type hypervector: np.ndarray
    :param density: Required density of the output hypervector, defaults to 0.5
    :type density: float, optional
    :return: Thinned hypervector
    :rtype: np.ndarray
    """
    num_nonzero = ceil(hypervector.size * density)
    indices = np.nonzero(hypervector)[0]
    if num_nonzero > indices.size:
        return hypervector
    random_indices = [
        indices[i]
        for i in np.random.choice(len(indices), size=num_nonzero, replace=False)
    ]
    hypervector.fill(0)
    hypervector[random_indices] = 1
    return hypervector


def SegmentedRandom(hypervector: np.ndarray, density: float = 0.5) -> np.ndarray:
    # TODO
    pass
