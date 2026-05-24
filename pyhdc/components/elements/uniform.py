from math import pi

import numpy as np


def UniformBipolar(dimensions: int, dtype: type = np.float32) -> np.ndarray:
    """UniformBipolar X∈R, X_i ~ U(-1,1)

    Uniform distribution of real numbers between -1 and 1

    :param dimensions: number of dimensions in the vector
    :type dimensions: int
    :return: hypervector
    :rtype: np.ndarray
    """
    return np.random.uniform(-1, 1, dimensions).astype(dtype)


def UniformAngles(dimensions: int, dtype: type = np.float32) -> np.ndarray:
    """UniformAngles θ∈R, θ_i ~ U(-pi, pi)

    Uniformly distributed angles from -pi to pi. Useful for a
    complex hypervector representation X∈C, X_i = e^(i*θ). In this
    case, the complex vector is assumed to be on the unit circle
    (length one) so we only need to store the real angle, θ, and
    the hypervector X can be computed as needed from the hypervector θ.

    :param dimensions: number of dimensions in the vector
    :type dimensions: int
    :return: hypervector
    :rtype: np.ndarray
    """
    return np.random.uniform(-pi, pi, dimensions).astype(dtype)
