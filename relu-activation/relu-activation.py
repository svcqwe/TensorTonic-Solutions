import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:      # если скаляр
        x = x.reshape(1)

    return np.maximum(0, x)