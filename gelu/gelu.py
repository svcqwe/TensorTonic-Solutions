import numpy as np
import math as m

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: scalar, list, or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    x = np.asarray(x, dtype=float)
    erf_ = np.vectorize(m.erf)
    return 0.5 * x * (1 + erf_(x / np.sqrt(2)))