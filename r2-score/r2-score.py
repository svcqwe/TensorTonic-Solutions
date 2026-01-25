import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if((len(y_true) != len(y_pred)) or (y_pred.ndim != 1) or (y_true.ndim != 1)):
      raise ValueError

    if(len(np.unique(np.concatenate([y_true, y_pred]))) == 1):
      return 1.0
    
    if len(np.unique(y_true)) == 1:
      return 0.0

    chislitel = np.sum(np.square(y_true - y_pred))
    znamenatel = np.sum(np.square(y_true - np.mean(y_true)))

    return 1 - (chislitel / znamenatel)

  

    