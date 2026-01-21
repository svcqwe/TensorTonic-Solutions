import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if((a.ndim != 1) or (b.ndim != 1) or (a.shape[0] != b.shape[0])):
        raise ValueError

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if((norm_a == 0) or (norm_b == 0)):
        return 0.0
    
    return (a @ b) / (norm_a * norm_b)
