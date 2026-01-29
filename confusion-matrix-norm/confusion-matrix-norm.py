import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute K×K confusion matrix with optional normalization.
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError

    if y_true.size == 0:
        K = num_classes or 0
        return np.zeros((K, K), dtype=float if normalize != 'none' else int)

    if num_classes is None:
        K = int(max(y_true.max(), y_pred.max()) + 1)
    else:
        K = int(num_classes)

    if (y_true < 0).any() or (y_true >= K).any():
        raise ValueError
    if (y_pred < 0).any() or (y_pred >= K).any():
        raise ValueError

    idx = y_true * K + y_pred
    cm = np.bincount(idx, minlength=K * K).reshape(K, K)

    if normalize == 'none':
        return cm.astype(int)

    cm = cm.astype(float)
    eps = 1e-12

    if normalize == 'true':
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        cm = cm / denom

    elif normalize == 'pred':
        denom = cm.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        cm = cm / denom

    elif normalize == 'all':
        denom = cm.sum()
        if denom == 0:
            denom = 1.0
        cm = cm / denom

    else:
        raise ValueError

    return cm
