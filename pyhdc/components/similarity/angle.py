import numpy as np


def AngleDistance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """AngleDistance Angle Distance of two hypervectors

    Calculates the average angular distance of two complex hypervectors
    of unit length. Results are normalized to the size of the hypervectors
    so it is always in the range [0,1].

    :param a: hypervector A
    :type a: np.ndarray
    :param b: hypervector B
    :type b: np.ndarray
    :return: similarity hypervector
    :rtype: np.ndarray
    """
    return np.sum(np.cos(a - b)) / a.size
    # cmp = (np.cos(a) - np.cos(b)) + ((np.sin(a) - np.sin(b)) * 1j)
    # sum = np.sum(np.cos(cmp))
    # return (sum / a.size).real
