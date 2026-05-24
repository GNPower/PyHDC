import numpy as np


def BernoulliBiploar(dimensions: int, dtype: type = np.int32) -> np.ndarray:
    """BernoulliBiploar X∈Z, X_i ~ B(0.5)*2 - 1

    Bernoulli distribution of bipolar numbers, either -1 or 1

    :param dimensions: number of dimensions in the vector
    :type dimensions: int
    :return: hypervector
    :rtype: np.ndarray
    """
    return np.random.choice([-1, 1], dimensions, p=[0.5, 0.5]).astype(dtype)


def BernoulliBinary(dimensions: int, dtype: type = np.int32) -> np.ndarray:
    """BernoulliBinary X∈{0,1}, X_i ~ B(0.5)

    Bernoulli distribution of binary numbers, either 0 or 1

    :param dimensions: number of dimensions in the vector
    :type dimensions: int
    :return: hypervector
    :rtype: np.ndarray
    """
    return np.random.binomial(size=dimensions, n=1, p=0.5).astype(dtype)
