import numpy as np

def knn_distance(X_train, X_test, k):

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)

    sorted_idx = np.argsort(dists, axis=1)

    result = np.full((n_test, k), -1, dtype=int)

    k_eff = min(k, n_train)
    result[:, :k_eff] = sorted_idx[:, :k_eff]

    return result
