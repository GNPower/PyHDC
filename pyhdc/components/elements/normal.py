from math import sqrt

import numpy as np


def NormalReal(dimensions: int, dtype: type = np.float32) -> np.ndarray:
    """NormalReal X∈R, X_i ~ N(0, 1/d)

    Normal distribution of real numbers with mean 0 and variance 1/d,
    where d is the dimension of the hypervector

    :param dimensions: number of dimensions in the vector
    :type dimensions: int
    :return: hypervector
    :rtype: np.ndarray
    """
    sigma = sqrt(1 / dimensions)
    return np.random.normal(0, sigma, dimensions).astype(dtype)
