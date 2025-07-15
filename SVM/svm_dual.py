import numpy as np
from scipy.optimize import minimize

def svm_dual(X_train, y_train, C):
    """
    Perform SVM in the dual domain using constrained optimization.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        C (float): Regularization parameter.

    Returns:
        tuple: (Lagrange multipliers, feature weights, bias)
    """
    n_samples, n_features = X_train.shape

    # Compute the kernel matrix Q
    Q = np.outer(y_train, y_train) * np.dot(X_train, X_train.T)

    # Define the dual objective function
    def objective(alpha):
        return 0.5 * np.dot(alpha.T, np.dot(Q, alpha)) - np.sum(alpha)

    # Define constraints
    constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y_train)}
    bounds = [(0, C) for _ in range(n_samples)]

    # Solve the dual problem using scipy.optimize.minimize
    result = minimize(
        fun=objective,
        x0=np.zeros(n_samples),  # Initial guess
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'
    )

    alpha = result.x

    # Recover feature weights
    w = np.sum((alpha * y_train)[:, None] * X_train, axis=0)

    # Recover bias
    support_indices = np.where((alpha > 1e-5) & (alpha < C))[0]
    b = np.mean([y_train[j] - np.dot(w, X_train[j]) for j in support_indices])

    return alpha, w, b

def svm_dual_gaussian(X_train, y_train, C, gamma):
    """
    Perform SVM in the dual domain using Gaussian kernel.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        C (float): Regularization parameter.
        gamma (float): Gaussian kernel parameter.

    Returns:
        tuple: (Lagrange multipliers, feature weights, bias)
    """
    n_samples = X_train.shape[0]

    # Compute the Gaussian kernel matrix Q
    Q = np.exp(-np.linalg.norm(X_train[:, None] - X_train[None, :], axis=2)**2 / gamma)
    Q *= np.outer(y_train, y_train)

    # Define the dual objective function
    def objective(alpha):
        return 0.5 * np.dot(alpha.T, np.dot(Q, alpha)) - np.sum(alpha)

    # Define constraints
    constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y_train)}
    bounds = [(0, C) for _ in range(n_samples)]

    # Solve the dual problem using scipy.optimize.minimize
    result = minimize(
        fun=objective,
        x0=np.zeros(n_samples),  # Initial guess
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'
    )

    alpha = result.x

    # Recover bias
    support_indices = np.where((alpha > 1e-5) & (alpha < C))[0]
    b = np.mean([y_train[j] - np.sum(alpha * y_train * Q[j]) for j in support_indices])

    return alpha, b

def predict_gaussian(X, X_train, y_train, alpha, b, gamma):
    """
    Predict using the Gaussian kernel SVM.

    Args:
        X (np.ndarray): Test features.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        alpha (np.ndarray): Lagrange multipliers.
        b (float): Bias term.
        gamma (float): Gaussian kernel parameter.

    Returns:
        np.ndarray: Predicted labels.
    """
    kernel = np.exp(-np.linalg.norm(X[:, None] - X_train[None, :], axis=2)**2 / gamma)
    return np.sign(np.dot(kernel, alpha * y_train) + b)