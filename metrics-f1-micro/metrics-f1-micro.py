import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if(len(y_true) != len(y_pred)):
        raise ValuError
    
    tp = 0
    for i in range(len(y_pred)):
        if(y_true[i] == y_pred[i]):
            tp += 1
        

    if(tp == len(y_pred)):
        return 1.0
    else:
        return (2 * tp) / (2 * tp + 2*(len(y_true) - tp))