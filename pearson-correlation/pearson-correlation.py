import numpy as np

def pearson_correlation(X):
    """
    Compute the Pearson correlation matrix from dataset X.
    """
    X = np.asarray(X, dtype=float)
    
    if ((X.ndim != 2) or (X.shape[0] < 2)):
        return None
    
    N, D = X.shape
    
    mu = X - np.mean(X, axis=0)
    cov = (mu.T @ mu) / (N - 1)
    stds = np.std(X, axis=0, ddof=1)
    
    zero_std = stds == 0
    
    denom = np.outer(stds, stds)
    
    R = cov / (denom + 1e-10)   #1e-10 для защиты от деления на ноль
    
    R[zero_std, :] = np.nan
    R[:, zero_std] = np.nan
    np.fill_diagonal(R, 1.0)
    
    return R
