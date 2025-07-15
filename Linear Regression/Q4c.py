import numpy as np
from data_loader import load_concrete_data, split_data
from gradient_descent import compute_cost

def compute_optimal_weights(X, y):
    """Compute optimal weights using the normal equation."""
    # Add column of ones for bias term
    X_tilde = np.hstack([X, np.ones((X.shape[0], 1))])
    
    # Compute optimal weights using normal equation
    # w_tilde = (X^T X)^(-1) X^T y
    w_tilde = np.linalg.inv(X_tilde.T @ X_tilde) @ X_tilde.T @ y
    
    # Split into bias and weights
    w = w_tilde[:-1]
    b = w_tilde[-1]
    
    return w, b

if __name__ == "__main__":
    zip_path = "../datasets/concrete+slump+test.zip"
    # Load data
    df = load_concrete_data(zip_path)
    X_train, y_train, X_test, y_test = split_data(df)

    # Compute optimal weights
    w_opt, b_opt = compute_optimal_weights(X_train, y_train)

    # Report results
    print("\nOptimal weight vector (analytical solution):")
    print(w_opt)
    print(f"Optimal bias term: {b_opt}")

    # Compute training cost
    train_cost = compute_cost(X_train, y_train, w_opt, b_opt)
    print(f"Training cost: {train_cost}")

    # Compute test cost
    test_cost = compute_cost(X_test, y_test, w_opt, b_opt)
    print(f"Test set cost: {test_cost}") 