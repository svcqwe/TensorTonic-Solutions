import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=int)

    if(len(y_train) == 0):
        raise ValueError
    if(len(X_test) == 0):
        return np.zeros(0, dtype=int)
    
    values, counts = np.unique(y_train, return_counts=True)
    most_freq = values[np.argmax(counts)]

    output = np.zeros((len(X_test), ), dtype=int)

    for i in range(len(output)):
        output[i] = most_freq
    
    return output
