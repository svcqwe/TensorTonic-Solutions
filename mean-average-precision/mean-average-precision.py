import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
    """
    ap_per_query = []
    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]

        if k is not None:
            y_true_sorted = y_true_sorted[:k]

        if np.sum(y_true_sorted) == 0:
            ap_per_query.append(0.0)
            continue

        cum_rel = np.cumsum(y_true_sorted)

        ranks = np.arange(1, len(y_true_sorted) + 1)
        precision = cum_rel / ranks

        ap = precision[y_true_sorted == 1].mean()
        ap_per_query.append(ap)

    map_value = float(np.mean(ap_per_query))

    return map_value, ap_per_query



