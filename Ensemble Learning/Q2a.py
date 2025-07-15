import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DecisionTree')))
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import load_bank_data
from decision_stumps import DecisionStump
from adaboost import AdaBoost

ZIP_PATH = '../datasets/bank-4.zip'
ATTRIBUTES = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"
]
LABEL = 'y'
NUMERIC_COLS = {"age", "balance", "day", "duration", "campaign", "pdays", "previous"}
DEPTHS = [1]  # Only stumps
T_MAX = 500

def compute_medians(data, numeric_cols):
    medians = {}
    for col in numeric_cols:
        values = [row[col] for row in data]
        medians[col] = float(np.median(values))
    return medians

def binarize_data(data, medians, numeric_cols):
    bin_data = []
    for row in data:
        new_row = row.copy()
        for col in numeric_cols:
            new_row[col] = ">" if row[col] > medians[col] else "<="
        bin_data.append(new_row)
    return bin_data

def label_to_pm1(y):
    return np.array([1 if v == 'yes' else -1 for v in y])

def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)

def main():
    train_data = load_bank_data(ZIP_PATH, split='train')
    test_data = load_bank_data(ZIP_PATH, split='test')
    medians = compute_medians(train_data, NUMERIC_COLS)
    train_bin = binarize_data(train_data, medians, NUMERIC_COLS)
    test_bin = binarize_data(test_data, medians, NUMERIC_COLS)
    y_train = label_to_pm1([row[LABEL] for row in train_bin])
    y_test = label_to_pm1([row[LABEL] for row in test_bin])

    X_train = [{k: v for k, v in row.items() if k != LABEL} for row in train_bin]
    X_test = [{k: v for k, v in row.items() if k != LABEL} for row in test_bin]

    train_errors = []
    test_errors = []
    stump_train_errors = []
    stump_test_errors = []

    ada = AdaBoost(T=T_MAX)
    ada.fit(X_train, y_train, ATTRIBUTES, T=T_MAX)

    # For each t, compute ensemble and stump errors
    for t in tqdm(range(1, len(ada.stumps) + 1), desc='Evaluating AdaBoost rounds'):
        # Ensemble prediction
        y_pred_train = ada.predict(X_train) if t == len(ada.stumps) else np.sign(sum(
            alpha * np.where(np.array(stump.predict(X_train)) == 1, 1, -1)
            for alpha, stump in zip(ada.alphas[:t], ada.stumps[:t])
        ))
        y_pred_test = ada.predict(X_test) if t == len(ada.stumps) else np.sign(sum(
            alpha * np.where(np.array(stump.predict(X_test)) == 1, 1, -1)
            for alpha, stump in zip(ada.alphas[:t], ada.stumps[:t])
        ))
        train_errors.append(error_rate(y_train, y_pred_train))
        test_errors.append(error_rate(y_test, y_pred_test))
        # Stump errors
        stump_pred_train = np.where(np.array(ada.stumps[t-1].predict(X_train)) == 1, 1, -1)
        stump_pred_test = np.where(np.array(ada.stumps[t-1].predict(X_test)) == 1, 1, -1)
        stump_train_errors.append(error_rate(y_train, stump_pred_train))
        stump_test_errors.append(error_rate(y_test, stump_pred_test))

    # Plot ensemble error vs T
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_errors)+1), train_errors, label='Train Error (Ensemble)')
    plt.plot(range(1, len(test_errors)+1), test_errors, label='Test Error (Ensemble)')
    plt.xlabel('Number of Boosting Rounds (T)')
    plt.ylabel('Error Rate')
    plt.title('AdaBoost: Ensemble Error vs T')
    plt.legend()
    plt.tight_layout()
    plt.savefig('adaboost_ensemble_error.png')
    plt.close()

    # Plot stump error vs T
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(stump_train_errors)+1), stump_train_errors, label='Train Error (Stump)')
    plt.plot(range(1, len(stump_test_errors)+1), stump_test_errors, label='Test Error (Stump)')
    plt.xlabel('Boosting Round (t)')
    plt.ylabel('Error Rate')
    plt.title('AdaBoost: Stump Error vs t')
    plt.legend()
    plt.tight_layout()
    plt.savefig('adaboost_stump_error.png')
    plt.close()

if __name__ == '__main__':
    main() 