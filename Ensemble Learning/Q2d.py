import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DecisionTree')))
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import load_bank_data
from random_forest import RandomForest

ZIP_PATH = '../datasets/bank-4.zip'
ATTRIBUTES = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"
]
LABEL = 'y'
NUMERIC_COLS = {"age", "balance", "day", "duration", "campaign", "pdays", "previous"}
T_MAX = 500
K_LIST = [2, 4, 6]


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

def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)

def main():
    train_data = load_bank_data(ZIP_PATH, split='train')
    test_data = load_bank_data(ZIP_PATH, split='test')
    medians = compute_medians(train_data, NUMERIC_COLS)
    train_bin = binarize_data(train_data, medians, NUMERIC_COLS)
    test_bin = binarize_data(test_data, medians, NUMERIC_COLS)
    y_train = np.array([row[LABEL] for row in train_bin])
    y_test = np.array([row[LABEL] for row in test_bin])

    X_train = [{k: v for k, v in row.items() if k != LABEL} for row in train_bin]
    X_test = [{k: v for k, v in row.items() if k != LABEL} for row in test_bin]

    results = {k: {'train': [], 'test': []} for k in K_LIST}

    for k in K_LIST:
        print(f'Random Forest with k={k} features per split')
        rf = RandomForest(T=T_MAX, k=k)
        rf.fit(X_train, y_train, ATTRIBUTES, T=T_MAX, k=k)
        # Collect predictions for each T
        train_votes = []
        test_votes = []
        for tree in tqdm(rf.trees, desc=f'Collecting votes for k={k}'):
            train_votes.append(tree.predict(X_train))
            test_votes.append(tree.predict(X_test))
        train_votes = np.array(train_votes)
        test_votes = np.array(test_votes)
        for T in tqdm(range(1, T_MAX + 1), desc=f'Ensemble size (k={k})'):
            # Majority vote among first T trees
            def majority_vote(votes):
                n = votes.shape[1]
                preds = []
                for i in range(n):
                    vals, counts = np.unique(votes[:T, i], return_counts=True)
                    preds.append(vals[np.argmax(counts)])
                return np.array(preds)
            y_pred_train = majority_vote(train_votes)
            y_pred_test = majority_vote(test_votes)
            results[k]['train'].append(error_rate(y_train, y_pred_train))
            results[k]['test'].append(error_rate(y_test, y_pred_test))

    # Plot ensemble error vs T for each k
    plt.figure(figsize=(10, 6))
    for k in K_LIST:
        plt.plot(range(1, T_MAX+1), results[k]['train'], label=f'Train Error (k={k})', linestyle='--')
        plt.plot(range(1, T_MAX+1), results[k]['test'], label=f'Test Error (k={k})')
    plt.xlabel('Number of Random Trees (T)')
    plt.ylabel('Error Rate')
    plt.title('Random Forest: Ensemble Error vs T for Different Feature Subset Sizes')
    plt.legend()
    plt.tight_layout()
    plt.savefig('random_forest_ensemble_error.png')
    plt.close()

if __name__ == '__main__':
    main() 