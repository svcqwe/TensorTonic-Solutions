import numpy as np

def silhouette_score(X, labels):
    """
    Compute the Silhouette Score for clustering.
    """

    X = np.asarray(X)
    labels = np.asarray(labels)

    n_samples = X.shape[0]

    diff = X[:, None, :] - X[None, :, :]

    dist = np.linalg.norm(diff, axis=2)

    same_cluster = labels[:, None] == labels[None, :]

    np.fill_diagonal(same_cluster, False)

    intra_sum = np.sum(dist * same_cluster, axis=1)

    intra_count = np.sum(same_cluster, axis=1)

    a = intra_sum / intra_count

    unique_labels = np.unique(labels)
    n_clusters = unique_labels.shape[0]

    mean_dist_to_clusters = np.zeros((n_samples, n_clusters))

    for idx, cluster in enumerate(unique_labels):
        cluster_mask = labels == cluster

        mean_dist_to_clusters[:, idx] = np.mean(
            dist[:, cluster_mask], axis=1
        )

    own_cluster_mask = labels[:, None] == unique_labels[None, :]

    mean_dist_to_clusters[own_cluster_mask] = np.inf

    b = np.min(mean_dist_to_clusters, axis=1)

    s = (b - a) / np.maximum(a, b)

    return np.mean(s)
