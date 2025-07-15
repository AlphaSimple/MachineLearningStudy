import zipfile
import pandas as pd
import numpy as np

def load_data(zip_path, train_file, test_file):
    """
    Load training and testing data from a zip file.

    Args:
        zip_path (str): Path to the zip file.
        train_file (str): Name of the training file inside the zip.
        test_file (str): Name of the testing file inside the zip.

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Load training data
        with z.open(train_file) as f:
            train_data = pd.read_csv(f, header=None)
        # Load testing data
        with z.open(test_file) as f:
            test_data = pd.read_csv(f, header=None)

    # Separate features and labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Convert labels from {0, 1} to {-1, 1}
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    return X_train, y_train, X_test, y_test