import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Binary mode
    if average == "binary":
        if pos_label is None:
            raise ValueError
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
        fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {"accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)}
    
    # Multi-class: get list of classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Confusion matrix
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    
    # TP, FP, FN per class
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    support = np.sum(cm, axis=1)  # количество примеров каждого класса
    
    if average == "micro":
        tp_sum = TP.sum()
        fp_sum = FP.sum()
        fn_sum = FN.sum()
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    elif average == "macro":
        precisions = [TP[i]/(TP[i]+FP[i]) if TP[i]+FP[i]>0 else 0.0 for i in range(len(classes))]
        recalls = [TP[i]/(TP[i]+FN[i]) if TP[i]+FN[i]>0 else 0.0 for i in range(len(classes))]
        f1s = [2*p*r/(p+r) if p+r>0 else 0.0 for p,r in zip(precisions, recalls)]
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

    elif average == "weighted":
        precisions = [TP[i]/(TP[i]+FP[i]) if TP[i]+FP[i]>0 else 0.0 for i in range(len(classes))]
        recalls = [TP[i]/(TP[i]+FN[i]) if TP[i]+FN[i]>0 else 0.0 for i in range(len(classes))]
        f1s = [2*p*r/(p+r) if p+r>0 else 0.0 for p,r in zip(precisions, recalls)]
        weights = support / support.sum()
        precision = np.sum(np.array(precisions) * weights)
        recall = np.sum(np.array(recalls) * weights)
        f1 = np.sum(np.array(f1s) * weights)

    else:
        raise ValueError
    
    return {"accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)}