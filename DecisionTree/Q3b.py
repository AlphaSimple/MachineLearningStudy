import os
import copy
import numpy as np
from collections import Counter
from data_loader import load_bank_data
from id3 import DecisionTree

# Constants
ZIP_PATH = '../datasets/bank-4.zip'
ATTRIBUTES = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"
]
LABEL = 'y'
CRITERIA = ['information_gain', 'majority_error', 'gini_index']
DEPTHS = list(range(1, 17))
NUMERIC_COLS = {"age", "balance", "day", "duration", "campaign", "pdays", "previous"}


def compute_medians(data, numeric_cols):
    medians = {}
    for col in numeric_cols:
        values = [row[col] for row in data]
        medians[col] = float(np.median(values))
    return medians

def binarize_data(data, medians, numeric_cols):
    bin_data = []
    for row in data:
        new_row = copy.deepcopy(row)
        for col in numeric_cols:
            new_row[col] = ">" if row[col] > medians[col] else "<="
        bin_data.append(new_row)
    return bin_data

def compute_majorities(data, attributes, numeric_cols):
    majorities = {}
    for attr in attributes:
        if attr in numeric_cols:
            continue
        values = [row[attr] for row in data if row[attr] != 'unknown']
        if values:
            majorities[attr] = Counter(values).most_common(1)[0][0]
        else:
            majorities[attr] = 'unknown'  # fallback if all are unknown
    return majorities

def fill_unknowns(data, majorities, attributes, numeric_cols):
    filled = []
    for row in data:
        new_row = copy.deepcopy(row)
        for attr in attributes:
            if attr in numeric_cols:
                continue
            if new_row[attr] == 'unknown':
                new_row[attr] = majorities[attr]
        filled.append(new_row)
    return filled

def error_rate(y_true, y_pred):
    return sum(yt != yp for yt, yp in zip(y_true, y_pred)) / len(y_true)

def run_experiments():
    train_data = load_bank_data(ZIP_PATH, split='train')
    test_data = load_bank_data(ZIP_PATH, split='test')
    majorities = compute_majorities(train_data, ATTRIBUTES, NUMERIC_COLS)
    train_filled = fill_unknowns(train_data, majorities, ATTRIBUTES, NUMERIC_COLS)
    test_filled = fill_unknowns(test_data, majorities, ATTRIBUTES, NUMERIC_COLS)
    medians = compute_medians(train_filled, NUMERIC_COLS)
    train_bin = binarize_data(train_filled, medians, NUMERIC_COLS)
    test_bin = binarize_data(test_filled, medians, NUMERIC_COLS)
    y_train = [row[LABEL] for row in train_bin]
    y_test = [row[LABEL] for row in test_bin]

    results = {criterion: {'train': [], 'test': []} for criterion in CRITERIA}

    for criterion in CRITERIA:
        for depth in DEPTHS:
            tree = DecisionTree(criterion=criterion, max_depth=depth)
            tree.fit(train_bin, ATTRIBUTES, label_name=LABEL)
            y_pred_train = tree.predict(train_bin)
            y_pred_test = tree.predict(test_bin)
            train_err = error_rate(y_train, y_pred_train)
            test_err = error_rate(y_test, y_pred_test)
            results[criterion]['train'].append(train_err)
            results[criterion]['test'].append(test_err)
            print(f"Criterion: {criterion}, Depth: {depth}, Train Error: {train_err:.4f}, Test Error: {test_err:.4f}")

    # Print summary table
    print("\nSummary Table: Average Prediction Errors")
    print(f"{'Criterion':<18}{'Depth':<8}{'Train Error':<15}{'Test Error':<15}")
    for criterion in CRITERIA:
        for i, depth in enumerate(DEPTHS):
            print(f"{criterion:<18}{depth:<8}{results[criterion]['train'][i]:<15.4f}{results[criterion]['test'][i]:<15.4f}")

if __name__ == "__main__":
    run_experiments() 