import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    if(len(fpr) != len(tpr)):
        raise ValueError
    
    auc = 0.0
    for i in range(len(fpr)-1):
        auc += 0.5 * (tpr[i] + tpr[i + 1]) * (fpr[i+1] - fpr[i])
    
    return auc