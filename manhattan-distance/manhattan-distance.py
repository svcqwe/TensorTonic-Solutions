import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if((x.ndim != 1) or (y.ndim != 1) or (x.ndim != y.ndim) or (len(x) != len(y))):
        raise ValueError
    
    return np.sum(np.fabs(x-y), axis=0, dtype=float)