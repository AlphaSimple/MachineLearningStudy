import numpy as np
from data_loader import load_data
from kernel_perceptron import kernel_perceptron, predict_kernel_perceptron

# Load data
zip_path = "../datasets/bank-note.zip"
train_file = "bank-note/train.csv"
test_file = "bank-note/test.csv"
X_train, y_train, X_test, y_test = load_data(zip_path, train_file, test_file)

# Hyperparameters
gamma_values = [0.1, 0.5, 1, 5, 100]
T = 10  # Number of epochs

results = {}

for gamma in gamma_values:
    # Train kernel Perceptron
    c = kernel_perceptron(X_train, y_train, gamma, T)

    # Compute training and test errors
    train_predictions = predict_kernel_perceptron(X_train, X_train, y_train, c, gamma)
    test_predictions = predict_kernel_perceptron(X_test, X_train, y_train, c, gamma)
    train_error = np.mean(train_predictions != y_train)
    test_error = np.mean(test_predictions != y_test)

    # Store results
    results[gamma] = {
        "train_error": train_error,
        "test_error": test_error
    }

# Print results
for gamma, result in results.items():
    print(f"gamma={gamma}: train_error={result['train_error']}, test_error={result['test_error']}")