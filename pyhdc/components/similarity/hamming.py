import numpy as np


def HammingDistance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """HammingDistance Hamming Distance of two vectors

    Counts the number of elements in the hypervectors where A[i] != B[i].
    Result is normalized to the size of the hypervector so it is always in
    the range [0,1].

    :param a: hypervector A
    :type a: np.ndarray
    :param b: hypervector B
    :type b: np.ndarray
    :return: similarity hypervector
    :rtype: np.ndarray
    """
    return 1 - (np.count_nonzero(a != b) / a.size)
