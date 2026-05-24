import numpy as np


def CosineSimilarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CosineSimilarity Cosine Similarity of two vectors

    cos(θ) = ( A dot B ) / ( norm(A) * norm(B) )


    :param a: hypervector A
    :type a: np.ndarray
    :param b: hypervector B
    :type b: np.ndarray
    :return: similarity hypervector
    :rtype: np.ndarray
    """
    return np.dot(a.astype(np.float32), b.astype(np.float32)) / (
        np.linalg.norm(a.astype(np.float32)) * np.linalg.norm(b.astype(np.float32))
    )
    # try:
    #     sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    #     return sim
    # except:
    #     print("@@@@@@ ERROR ERROR ERROR @@@@@@")
    #     print(f"sim: {sim}")
    #     print(f"a: {a}")
    #     print(f"b: {b}")
    #     return -100
