import numpy as np

def compute_cost(X, y, w, b):
    """Compute cost function"""
    y_pred = X @ w + b
    cost = (1/2) * np.sum((y_pred - y)**2)
    return cost

def compute_gradient(X, y, w, b):
    """Compute gradient of the cost function with respect to w and b."""    
    y_pred = X @ w + b
    error = y - y_pred
    grad_w = -X.T @ error
    grad_b = -np.sum(error)
    return grad_w, grad_b

def batch_gradient_descent(X, y, r, tol=1e-6, max_iter=10000):
    """Run batch gradient descent. Returns weight vector, bias, cost history, and number of steps."""
    w = np.zeros(X.shape[1])
    b = 0
    cost_history = []
    for i in range(max_iter):
        grad_w, grad_b = compute_gradient(X, y, w, b)
        w_new = w - r * grad_w
        b_new = b - r * grad_b
        cost = compute_cost(X, y, w_new, b_new)
        cost_history.append(cost)
        if np.linalg.norm(w_new - w) < tol:
            return w_new, b_new, cost_history, i+1
        w, b = w_new, b_new
    return w, b, cost_history, max_iter

def stochastic_gradient_descent(X, y, r, tol=1e-6, max_iter=10000):
    """Run stochastic gradient descent. Returns weight vector, bias, cost history, and number of updates."""
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    cost_history = []
    prev_cost = float('inf')
    for i in range(max_iter):
        # Randomly sample an index
        np.random.seed(42)  # seed for reproducibility
        idx = np.random.randint(m)
        X_i = X[idx, :].reshape(1, -1)
        y_i = y[idx]
        # Compute gradient for the sampled point
        grad_w, grad_b = compute_gradient(X_i, y_i, w, b)
        # Update weights
        w -= r * grad_w
        b -= r * grad_b
        # Compute cost for the entire dataset after this update
        current_cost = compute_cost(X, y, w, b)
        cost_history.append(current_cost)
        # Check convergence based on cost function values
        if i > 0 and abs(current_cost - prev_cost) < tol:
            return w, b, cost_history, i+1
        prev_cost = current_cost
    return w, b, cost_history, max_iter