import random
from math import ceil, sqrt

import numpy as np


def BernoulliSparse(
    dimensions: int, dtype: type = np.int32, probability: float = None
) -> np.ndarray:
    """BernoulliSparse X∈{0,1}, X_i ~ B(p << 1)

    Bernoulli distribution of binary numbers, either 0 or 1 with variable
    probability. If undefined probability is given as p = 1/sqrt(dimensions),
    which creates a sparsely populated array

    :param dimensions: number of dimensions in the vector
    :type dimensions: int
    :param probability: probability of the Bernoulli distribution, defaults to 1/sqrt(d)
    :type probability: float, optional
    :return: hypervector
    :rtype: np.ndarray
    """
    if probability is None:
        probability = 1 / sqrt(dimensions)
    return np.random.binomial(size=dimensions, n=1, p=probability).astype(dtype)


def SparseSegmented(
    dimensions: int, dtype: type = np.int32, probability: float = None
) -> np.ndarray:
    """SparseSegmented X∈{0,1}, X_i ~ B(p << 1)

    Sparsely segmented binary numbers, either 0 or 1 with variable
    probability. If undefined probability is given as p = 1/sqrt(dimensions).
    Hypervector is split into s, dimensions * probability, segments. Each segment
    populated with exactly 1 non-zero element uniformly distributed throughout the
    segment.

    NOTE: Since s = dimensions * probability is not garunteed to evenly divide into
    the hypervector dimensions, the segment, s, is rounded up and the final
    hypervector trimmed. This means the non-zero value in the last segment may be
    trimmed and not be present in the final hypervector.

    :param dimensions: number of dimensions in the vector
    :type dimensions: int
    :param probability: probability of the Bernoulli distribution, defaults to 1/sqrt(d)
    :type probability: float, optional
    :return: hypervector
    :rtype: np.ndarray
    """
    if probability is None:
        probability = 1 / sqrt(dimensions)
    s = ceil(dimensions * probability)
    seg_d = ceil(dimensions / s)
    hypervector = np.zeros(0, dtype=np.int8)
    for i in range(0, s):
        vector = np.zeros(seg_d, dtype=np.int8)
        index = random.randrange(seg_d)
        vector[index] = 1
        hypervector = np.append(hypervector, vector)
    return hypervector[:dimensions].astype(dtype)
