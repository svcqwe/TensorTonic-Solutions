import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """

    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)

    n_left = len(y_left)
    n_right = len(y_right)

    if n_left == 0 and n_right == 0:
        return 0.0

    def gini(y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs ** 2)

    if n_left == 0:
        return gini(y_right)
    if n_right == 0:
        return gini(y_left)

    values, counts = np.unique(y_left, return_counts=True)
    left_sum = 0.0
    for i in range(len(values)):
        left_sum += np.square(counts[i] / np.sum(counts))

    values, counts = np.unique(y_right, return_counts=True)
    right_sum = 0.0
    for i in range(len(values)):
        right_sum += np.square(counts[i] / np.sum(counts))

    Nleft_N = n_left / (n_left + n_right)
    Nright_N = n_right / (n_left + n_right)

    gini_split = Nleft_N * (1 - left_sum) + Nright_N * (1 - right_sum)

    return gini_split