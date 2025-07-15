import numpy as np

def standard_perceptron(X_train, y_train, T, r=1.0):
    """
    Standard Perceptron algorithm.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        T (int): Maximum number of epochs.
        r (float): Learning rate.

    Returns:
        np.ndarray: Learned weight vector.
    """
    n_features = X_train.shape[1]
    w = np.zeros(n_features)

    for epoch in range(T):
        # Shuffle the data
        rng = np.random.default_rng(seed=epoch)
        indices = rng.permutation(len(y_train))
        X_train, y_train = X_train[indices], y_train[indices]

        for x_i, y_i in zip(X_train, y_train):
            if y_i * np.dot(w, x_i) <= 0:
                w += r * y_i * x_i

    return w

def voted_perceptron(X_train, y_train, T, r=1.0):
    """
    Voted Perceptron algorithm.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        T (int): Maximum number of epochs.
        r (float): Learning rate.

    Returns:
        list: List of tuples (weight vector, count).
    """
    n_features = X_train.shape[1]
    w = np.zeros(n_features)
    weights = []
    count = 0

    for epoch in range(T):
        for x_i, y_i in zip(X_train, y_train):
            if y_i * np.dot(w, x_i) <= 0:
                weights.append((w.copy(), count))
                w += r * y_i * x_i
                count = 1
            else:
                count += 1

    weights.append((w.copy(), count))
    return weights

def averaged_perceptron(X_train, y_train, T, r=1.0):
    """
    Averaged Perceptron algorithm.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        T (int): Maximum number of epochs.
        r (float): Learning rate.

    Returns:
        np.ndarray: Averaged weight vector.
    """
    n_features = X_train.shape[1]
    w = np.zeros(n_features)
    a = np.zeros(n_features)

    for epoch in range(T):
        for x_i, y_i in zip(X_train, y_train):
            if y_i * np.dot(w, x_i) <= 0:
                w += r * y_i * x_i
            a += w

    return a