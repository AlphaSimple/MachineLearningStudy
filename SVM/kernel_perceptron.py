import numpy as np

def gaussian_kernel(x_i, x_j, gamma):
    """
    Compute the Gaussian kernel between two points.

    Args:
        x_i (np.ndarray): First point.
        x_j (np.ndarray): Second point.
        gamma (float): Kernel parameter.

    Returns:
        float: Kernel value.
    """
    return np.exp(-np.linalg.norm(x_i - x_j)**2 / gamma)

def kernel_perceptron(X_train, y_train, gamma, T):
    """
    Train the kernel Perceptron algorithm using Gaussian kernel.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        gamma (float): Kernel parameter.
        T (int): Number of epochs.

    Returns:
        np.ndarray: Mistake counts (c_i) for all training examples.
    """
    n_samples = X_train.shape[0]
    c = np.zeros(n_samples)  # Initialize mistake counts

    for t in range(T):
        for i in range(n_samples):
            # Compute prediction
            prediction = np.sign(
                sum(c[j] * y_train[j] * gaussian_kernel(X_train[j], X_train[i], gamma) for j in range(n_samples))
            )

            # Update mistake count if prediction is incorrect
            if prediction != y_train[i]:
                c[i] += 1

    return c

def predict_kernel_perceptron(X, X_train, y_train, c, gamma):
    """
    Predict using the kernel Perceptron algorithm.

    Args:
        X (np.ndarray): Test features.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        c (np.ndarray): Mistake counts.
        gamma (float): Kernel parameter.

    Returns:
        np.ndarray: Predicted labels.
    """
    kernel = np.exp(-np.linalg.norm(X[:, None] - X_train[None, :], axis=2)**2 / gamma)
    return np.sign(np.dot(kernel, c * y_train))