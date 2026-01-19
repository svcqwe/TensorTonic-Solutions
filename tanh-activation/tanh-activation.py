import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asarray(x, dtype=float)

    if(x.ndim == 0):
        x = x.reshape(1)

    x = np.clip(x, -1000, 1000)

    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))