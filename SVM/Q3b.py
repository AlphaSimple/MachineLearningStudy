import numpy as np
from data_loader import load_data
from svm_dual import svm_dual_gaussian, predict_gaussian

# Load data
zip_path = "../datasets/bank-note.zip"
train_file = "bank-note/train.csv"
test_file = "bank-note/test.csv"
X_train, y_train, X_test, y_test = load_data(zip_path, train_file, test_file)

# Hyperparameters
C_values = [100 / 873, 500 / 873, 700 / 873]
gamma_values = [0.1, 0.5, 1, 5, 100]

results = {}

for C in C_values:
    for gamma in gamma_values:
        # Train SVM with Gaussian kernel
        alpha, b = svm_dual_gaussian(X_train, y_train, C, gamma)

        # Compute training and test errors
        train_predictions = predict_gaussian(X_train, X_train, y_train, alpha, b, gamma)
        test_predictions = predict_gaussian(X_test, X_train, y_train, alpha, b, gamma)
        train_error = np.mean(train_predictions != y_train)
        test_error = np.mean(test_predictions != y_test)

        # Store results
        results[(C, gamma)] = {
            "train_error": train_error,
            "test_error": test_error
        }

# Print results
for (C, gamma), result in results.items():
    print(f"C={C}, gamma={gamma}: train_error={result['train_error']}, test_error={result['test_error']}")

# Identify the best combination
# Find all combinations with the minimum test error
min_test_error = min(result['test_error'] for result in results.values())
best_combinations = [(C, gamma, result['test_error']) for (C, gamma), result in results.items() if result['test_error'] == min_test_error]

# Print all best combinations
print("Best combinations:")
for C, gamma, test_error in best_combinations:
    print(f"C={C}, gamma={gamma} with test_error={test_error}")