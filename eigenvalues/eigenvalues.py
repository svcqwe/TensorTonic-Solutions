import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    A = np.asarray(matrix)
    if((A.ndim != 2) or (A.shape[0] != A.shape[1])):
        return None

    return np.linalg.eigvals(A)
    