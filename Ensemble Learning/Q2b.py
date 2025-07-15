import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DecisionTree')))
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import load_bank_data
from id3 import DecisionTree
from bagged_trees import BaggedTrees
from collections import Counter

ZIP_PATH = '../datasets/bank-4.zip'
ATTRIBUTES = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"
]
LABEL = 'y'
NUMERIC_COLS = {"age", "balance", "day", "duration", "campaign", "pdays", "previous"}
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

def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)

def majority_vote(preds):
    # preds: shape (T, n)
    n = preds.shape[1]
    final = []
    for i in range(n):
        count = Counter(preds[:, i])
        final.append(count.most_common(1)[0][0])
    return np.array(final)

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

    # Train all T_MAX trees once
    n = len(X_train)
    all_trees = []
    train_preds = []
    test_preds = []
    print('Training all bagged trees...')
    for t in tqdm(range(T_MAX), desc='Training trees'):
        idxs = np.random.choice(n, n, replace=True)
        X_boot = [X_train[i] for i in idxs]
        y_boot = [y_train[i] for i in idxs]
        data_boot = [dict(x) for x in X_boot]
        for i, label in enumerate(y_boot):
            data_boot[i]['label'] = label
        tree = BaggedTrees(T=1)  # Use a single tree for this round
        dtree = DecisionTree(criterion='information_gain', max_depth=None)
        dtree.fit(data_boot, ATTRIBUTES, label_name='label')
        all_trees.append(dtree)
        train_preds.append(dtree.predict(X_train))
        test_preds.append(dtree.predict(X_test))
    train_preds = np.array(train_preds)  # shape (T_MAX, n_train)
    test_preds = np.array(test_preds)    # shape (T_MAX, n_test)

    train_errors = []
    test_errors = []
    print('Evaluating ensemble predictions...')
    for T in tqdm(range(1, T_MAX + 1), desc='Bagging rounds'):
        y_pred_train = majority_vote(train_preds[:T, :])
        y_pred_test = majority_vote(test_preds[:T, :])
        train_errors.append(error_rate(y_train, y_pred_train))
        test_errors.append(error_rate(y_test, y_pred_test))

    # Plot ensemble error vs T
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, T_MAX+1), train_errors, label='Train Error (Bagged Ensemble)')
    plt.plot(range(1, T_MAX+1), test_errors, label='Test Error (Bagged Ensemble)')
    plt.xlabel('Number of Bagged Trees (T)')
    plt.ylabel('Error Rate')
    plt.title('Bagged Trees: Ensemble Error vs T')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bagged_trees_ensemble_error.png')
    plt.close()

if __name__ == '__main__':
    main() 