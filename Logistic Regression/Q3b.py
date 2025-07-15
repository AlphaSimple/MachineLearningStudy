import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_ml_gradient(X, y, w):
    z = y * (X @ w)
    coeff = (1 - sigmoid(z)) * y
    grad = -np.mean(coeff[:, None] * X, axis=0)
    return grad

def train_ml_sgd(X, y, gamma_0, d, T):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    objective_vals = []
    updates = 0

    for epoch in range(T):
        np.random.seed(epoch+1)
        indices = np.random.permutation(n_samples)
        for i in indices:
            gamma_t = gamma_0 / (1 + (gamma_0 / d) * updates)
            grad = compute_ml_gradient(X[i:i+1], y[i:i+1], w)
            w -= gamma_t * grad

            z = y * (X @ w)
            log_likelihood = np.sum(-np.log(sigmoid(z)))
            objective_vals.append(log_likelihood)
            updates += 1
    return w, objective_vals

def evaluate(X, y, w):
    preds = np.sign(X @ w)
    error = np.mean(preds != y)
    return error

if __name__ == "__main__":
    zip_path = '../datasets/bank-note.zip'
    train_file = 'bank-note/train.csv'
    test_file = 'bank-note/test.csv'
    X_train, y_train, X_test, y_test = load_data(zip_path, train_file, test_file)

    # Add bias term
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    gamma_0 = 0.01
    d = 1
    T = 100

    w, obj_vals = train_ml_sgd(X_train, y_train, gamma_0, d, T)
    train_err = evaluate(X_train, y_train, w)
    test_err = evaluate(X_test, y_test, w)
    print(f"ML Estimation | Train Error: {train_err:.4f} | Test Error: {test_err:.4f}")

    plt.plot(obj_vals)
    plt.title("MLE over SGD Updates")
    plt.xlabel("Update steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Q3b.png')
    plt.show()
