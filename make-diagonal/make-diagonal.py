import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    A = np.zeros((len(v), len(v)), dtype=float)

    for i in range(len(A)):
        A[i, i] = v[i]

    return A

