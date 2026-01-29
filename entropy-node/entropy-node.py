import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y, dtype=float)

    if((len(np.unique(y)) == 1) or (len(y) == 0)):
        return 0.0
    
    values, counts = np.unique(y, return_counts=True)

    H = 0.0
    for i in range(len(np.unique(y))):
        p = counts[i] / len(y)
        H += p * np.log2(p)
    
    return -H

    