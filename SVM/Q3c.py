import numpy as np
from data_loader import load_data
from svm_dual import svm_dual_gaussian

# Load data
zip_path = "../datasets/bank-note.zip"
train_file = "bank-note/train.csv"
test_file = "bank-note/test.csv"
X_train, y_train, X_test, y_test = load_data(zip_path, train_file, test_file)

# Hyperparameters
C_values = [100 / 873, 500 / 873, 700 / 873]
gamma_values = [0.01, 0.1, 0.5, 1, 5, 100]

results = {}

# Analyze support vectors
for C in C_values:
    results[C] = {}
    for gamma in gamma_values:
        # Train SVM with Gaussian kernel
        alpha, b = svm_dual_gaussian(X_train, y_train, C, gamma)

        # Identify support vectors
        support_indices = np.where((alpha > 1e-5) & (alpha < C))[0]
        num_support_vectors = len(support_indices)

        # Store results
        results[C][gamma] = {
            "support_indices": support_indices,
            "num_support_vectors": num_support_vectors
        }

# Print number of support vectors for each combination of C and gamma
for C in C_values:
    print(f"C={C}:")
    for gamma in gamma_values:
        num_support_vectors = results[C][gamma]["num_support_vectors"]
        print(f"  gamma={gamma}: num_support_vectors={num_support_vectors}")

# Overlap analysis for C = 500 / 873
C_target = 500 / 873
print(f"\nOverlap analysis for C={C_target}:")
for i in range(len(gamma_values) - 1):
    gamma1 = gamma_values[i]
    gamma2 = gamma_values[i + 1]
    support_indices_gamma1 = results[C_target][gamma1]["support_indices"]
    support_indices_gamma2 = results[C_target][gamma2]["support_indices"]

    overlap = len(np.intersect1d(support_indices_gamma1, support_indices_gamma2))
    print(f"  Overlap between gamma={gamma1} and gamma={gamma2}: {overlap}")