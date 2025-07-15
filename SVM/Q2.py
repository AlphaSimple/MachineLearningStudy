import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data
from svm_sgd import svm_sgd

# Load data
zip_path = '../datasets/bank-note.zip'
train_file = 'bank-note/train.csv'
test_file = 'bank-note/test.csv'
X_train, y_train, X_test, y_test = load_data(zip_path, train_file, test_file)

# Hyperparameters
C_values = [100 / 873, 500 / 873, 700 / 873]
gamma_0_values = [0.5]
a_values = [0.001]
T = 100

results = {}

for C in C_values:
    for gamma_0 in gamma_0_values:
        for a in a_values:
            for schedule_type in ['part-a', 'part-b']:
                # Train SVM
                w, objective_values = svm_sgd(X_train, y_train, C, gamma_0, a, T, schedule_type)

                # Compute training and test errors
                train_predictions = np.sign(np.dot(X_train, w))
                test_predictions = np.sign(np.dot(X_test, w))
                train_error = np.mean(train_predictions != y_train)
                test_error = np.mean(test_predictions != y_test)

                # Store results
                key = (C, gamma_0, a, schedule_type)
                results[key] = {
                    "train_error": train_error,
                    "test_error": test_error,
                    "objective_values": objective_values,
                    "weights": w
                }

# Create subplots for learning rate schedules
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for i, schedule_type in enumerate(['part-a', 'part-b']):
    for key, result in results.items():
        if key[3] == schedule_type:
            axes[i].plot(result["objective_values"], label=f"C={key[0]:.2f}, $\\gamma_0$={key[1]}, a={key[2]}")
    axes[i].set_xlabel("Epochs")
    axes[i].set_ylabel("Objective Function Value")
    axes[i].set_yscale("log")
    axes[i].legend()
    axes[i].set_title(f"Objective Function Convergence ({schedule_type})")
    # Add grid lines to subplots
    axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('obj_func_plots.png')
plt.show()

# Compare results
for C in C_values:
    print(f"Results for C={C}:")
    
    weights_type1 = results[(C, gamma_0_values[0], a_values[0], 'part-a')]["weights"]
    weights_type2 = results[(C, gamma_0_values[0], a_values[0], 'part-b')]["weights"]
    train_error_type1 = results[(C, gamma_0_values[0], a_values[0], 'part-a')]["train_error"]
    train_error_type2 = results[(C, gamma_0_values[0], a_values[0], 'part-b')]["train_error"]
    test_error_type1 = results[(C, gamma_0_values[0], a_values[0], 'part-a')]["test_error"]
    test_error_type2 = results[(C, gamma_0_values[0], a_values[0], 'part-b')]["test_error"]
    
    print(f"Schedule part-a: train_error={train_error_type1}, test_error={test_error_type1}")
    print(f"Schedule part-b: train_error={train_error_type2}, test_error={test_error_type2}")
    
    print(f"Schedule part-a weights: {weights_type1}")
    print(f"Schedule part-b weights: {weights_type2}")
    # print(f"Weight difference: {np.linalg.norm(weights_type1 - weights_type2)}\n")