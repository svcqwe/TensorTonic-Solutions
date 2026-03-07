import math as m

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    if(len(y_true) != len(y_pred)):
        raise ValueError
    
    loss = []
    for i in range(len(y_true)):
        p = min(max(y_pred[i], eps), 1 - eps)
        loss.append(-(y_true[i] * m.log(p) + (1 - y_true[i]) * m.log(1 - p)))

    return loss