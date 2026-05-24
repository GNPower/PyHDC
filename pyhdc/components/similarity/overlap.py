import numpy as np


def Overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Overlap Overlap of two vectors

    Counts the number of elements in the hyypervectors where both A[i] and
    B[i] are 1. Useful for sparce hypervectors. Result is normalized to the
    number of nonzero elements in the reference (second) hypervector so it
    is always in the range [0,1]. Normalizing against the second hypervector
    means the best results will be acheived from passing the bundled
    hypervector as A and the reference hypervector as B.

    :param a: hypervector A
    :type a: np.ndarray
    :param b: hypervector B
    :type b: np.ndarray
    :return: similarity hypervector
    :rtype: np.ndarray
    """
    return np.count_nonzero(np.logical_and(a == b, a == 1)) / max(np.sum(b), 1)
