import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    return np.sqrt(np.sum(np.square(x-y), axis=0))
