import numpy as np

def svm_sgd(X_train, y_train, C, gamma_0, a, T, schedule_type):
    """
    Perform SVM in the primal domain using stochastic sub-gradient descent.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        C (float): Regularization parameter.
        gamma_0 (float): Initial learning rate.
        a (float): Parameter for learning rate schedule.
        T (int): Maximum number of epochs.
        schedule_type (str): Type of learning rate schedule ('type1' or 'type2').

    Returns:
        tuple: (final weights, objective function values)
    """
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)  # Initialize weights
    objective_values = []

    for epoch in range(1, T + 1):
        # Shuffle the training data
        np.random.seed(epoch+1)  # Set random seed for reproducibility
        indices = np.random.permutation(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(n_samples):
            x_i = X_train[i]
            y_i = y_train[i]

            # Compute learning rate
            if schedule_type == 'part-a':
                gamma_t = gamma_0 / (1 + (gamma_0 / a) * epoch)
            elif schedule_type == 'part-b':
                gamma_t = gamma_0 / (1 + epoch)
            else:
                raise ValueError("Invalid schedule_type. Use 'part-a' or 'part-b'.")

            # Update weights based on hinge loss condition
            if y_i * np.dot(w, x_i) <= 1:
                w = w - gamma_t * w + gamma_t * C * n_samples * y_i * x_i
            else:
                w = (1 - gamma_t) * w

        # Compute objective function value
        hinge_loss = np.maximum(0, 1 - y_train * np.dot(X_train, w)).sum()
        regularization = 0.5 * np.dot(w, w)
        objective_values.append(C * hinge_loss + regularization)

    return w, objective_values