import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DecisionTree')))
import numpy as np
from tqdm import tqdm
from data_loader import load_bank_data
from id3 import DecisionTree
from bagged_trees import BaggedTrees

ZIP_PATH = '../datasets/bank-4.zip'
ATTRIBUTES = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"
]
LABEL = 'y'
NUMERIC_COLS = {"age", "balance", "day", "duration", "campaign", "pdays", "previous"}
N_RUNS = 100
N_SAMPLE = 1000
N_TREES = 500


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

def main():
    train_data = load_bank_data(ZIP_PATH, split='train')
    test_data = load_bank_data(ZIP_PATH, split='test')
    medians = compute_medians(train_data, NUMERIC_COLS)
    train_bin = binarize_data(train_data, medians, NUMERIC_COLS)
    test_bin = binarize_data(test_data, medians, NUMERIC_COLS)
    y_test = label_to_pm1([row[LABEL] for row in test_bin])
    X_test = [{k: v for k, v in row.items() if k != LABEL} for row in test_bin]

    n_test = len(X_test)
    single_tree_preds = np.zeros((N_RUNS, n_test))
    bagged_preds = np.zeros((N_RUNS, n_test))

    print('Running bias-variance experiment...')
    for run in tqdm(range(N_RUNS), desc='Runs'):
        idxs = np.random.choice(len(train_bin), N_SAMPLE, replace=False)
        sample = [train_bin[i] for i in idxs]
        y_sample = [row[LABEL] for row in sample]
        X_sample = [{k: v for k, v in row.items() if k != LABEL} for row in sample]
        y_sample_pm1 = label_to_pm1(y_sample)
        # Train 500 bagged trees
        all_trees = []
        for t in range(N_TREES):
            boot_idxs = np.random.choice(N_SAMPLE, N_SAMPLE, replace=True)
            X_boot = [X_sample[i] for i in boot_idxs]
            y_boot = [y_sample_pm1[i] for i in boot_idxs]
            data_boot = [dict(x) for x in X_boot]
            for i, label in enumerate(y_boot):
                data_boot[i]['label'] = label
            tree = DecisionTree(criterion='information_gain', max_depth=None)
            tree.fit(data_boot, ATTRIBUTES, label_name='label')
            all_trees.append(tree)
        # Single tree: first tree in each run
        single_tree_preds[run, :] = np.array([1 if p == 'yes' or p == 1 else -1 for p in all_trees[0].predict(X_test)])
        # Bagged: majority vote of all 500 trees
        all_tree_preds = np.array([[1 if p == 'yes' or p == 1 else -1 for p in tree.predict(X_test)] for tree in all_trees])
        bagged_pred = np.sign(np.sum(all_tree_preds, axis=0))
        bagged_pred[bagged_pred == 0] = 1  # break ties as +1
        bagged_preds[run, :] = bagged_pred

    # Bias-variance for single tree
    single_means = np.mean(single_tree_preds, axis=0)
    single_bias2 = np.mean((single_means - y_test) ** 2)
    single_var = np.mean(np.var(single_tree_preds, axis=0, ddof=1))
    single_sqerr = single_bias2 + single_var
    # Bias-variance for bagged ensemble
    bagged_means = np.mean(bagged_preds, axis=0)
    bagged_bias2 = np.mean((bagged_means - y_test) ** 2)
    bagged_var = np.mean(np.var(bagged_preds, axis=0, ddof=1))
    bagged_sqerr = bagged_bias2 + bagged_var

    print('Single tree:   Bias^2 = {:.4f}, Variance = {:.4f}, Squared Error = {:.4f}'.format(single_bias2, single_var, single_sqerr))
    print('Bagged trees:  Bias^2 = {:.4f}, Variance = {:.4f}, Squared Error = {:.4f}'.format(bagged_bias2, bagged_var, bagged_sqerr))

if __name__ == '__main__':
    main() 