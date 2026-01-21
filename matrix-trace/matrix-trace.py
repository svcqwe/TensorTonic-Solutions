import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    res = 0.0
    A = np.asarray(A, dtype=float)
    
    if((A.ndim != 2) or (A.shape[0] != A.shape[1])):
        raise ValueError

    for i in range(A.shape[0]):
        res += A[i,i]

    return res