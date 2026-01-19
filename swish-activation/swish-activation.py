import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    
    x = np.asarray(x, dtype=float)

    if x.ndim == 0:
        x = x.reshape(1)
        
    z = np.clip(x, -1000, 1000)
    _z = np.clip(-x, -1000, 1000)
    
    
    s = np.where(x >= 0, 1 / (1 + np.exp(_z)), np.exp(z) / (1 + np.exp(z)))
    return x * s