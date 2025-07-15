import os
from data_loader import load_car_data
from id3 import DecisionTree

# Constants
ZIP_PATH = '../datasets/car-4.zip'
ATTRIBUTES = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
LABEL = 'label'
CRITERIA = ['information_gain', 'majority_error', 'gini_index']
DEPTHS = list(range(1, 7))


def error_rate(y_true, y_pred):
    return sum(yt != yp for yt, yp in zip(y_true, y_pred)) / len(y_true)


def run_experiments():
    train_data = load_car_data(ZIP_PATH, split='train')
    test_data = load_car_data(ZIP_PATH, split='test')
    y_train = [row[LABEL] for row in train_data]
    y_test = [row[LABEL] for row in test_data]

    results = {criterion: {'train': [], 'test': []} for criterion in CRITERIA}

    for criterion in CRITERIA:
        for depth in DEPTHS:
            tree = DecisionTree(criterion=criterion, max_depth=depth)
            tree.fit(train_data, ATTRIBUTES, label_name=LABEL)
            y_pred_train = tree.predict(train_data)
            y_pred_test = tree.predict(test_data)
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