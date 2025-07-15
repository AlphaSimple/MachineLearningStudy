import numpy as np
from data_loader import load_data
from svm_dual import svm_dual
from svm_sgd import svm_sgd

# Load data
zip_path = "../datasets/bank-note.zip"
train_file = 'bank-note/train.csv'
test_file = 'bank-note/test.csv'
X_train, y_train, X_test, y_test = load_data(zip_path, train_file, test_file)

# Hyperparameters
C_values = [100 / 873, 500 / 873, 700 / 873]

results_dual = {}
results_primal = {}

for C in C_values:
    # Train SVM in the dual domain
    alpha, w_dual, b_dual = svm_dual(X_train, y_train, C)

    # Compute training and test errors for dual domain
    train_predictions_dual = np.sign(np.dot(X_train, w_dual) + b_dual)
    test_predictions_dual = np.sign(np.dot(X_test, w_dual) + b_dual)
    train_error_dual = np.mean(train_predictions_dual != y_train)
    test_error_dual = np.mean(test_predictions_dual != y_test)

    results_dual[C] = {
        "weights": w_dual,
        "bias": b_dual,
        "train_error": train_error_dual,
        "test_error": test_error_dual
    }

# Compare results
for C in C_values:
    print(f"Results for C={C}:")
    print(f"Dual SVM: train_error={results_dual[C]['train_error']}, test_error={results_dual[C]['test_error']}")
    # print(f"Primal SVM: train_error={results_primal[C]['train_error']}, test_error={results_primal[C]['test_error']}")
    print(f"Weight vector (dual): {results_dual[C]['weights']}")
    # print(f"Weight difference: {np.linalg.norm(results_dual[C]['weights'] - results_primal[C]['weights'])}")
    print(f"Bias (dual): {results_dual[C]['bias']}")
    print("---")