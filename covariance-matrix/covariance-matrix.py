import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.asarray(X, dtype=float)
    
    if((X.ndim != 2) or (X.shape[0] < 2)):
        return None

    N, D = X.shape
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    cov_matrix = (X_centered.T @ X_centered) / (N - 1)

    return cov_matrix