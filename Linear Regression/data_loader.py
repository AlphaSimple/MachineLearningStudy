import zipfile
import pandas as pd
import io

def load_concrete_data(zip_path):
    """Load the concrete slump data from the zip file and return as pandas DataFrame."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open('slump_test.data') as f:
            # The first line is the header, but the second line is a continuation of the header
            lines = f.read().decode('utf-8').splitlines()
            header = lines[0].strip().split(',')
            # Remove any empty or whitespace-only header entries
            header = [h.strip() for h in header if h.strip()]
            # Data starts from line 2 (index 1)
            data = '\n'.join(lines[1:])
            df = pd.read_csv(io.StringIO(data), names=header)
    return df

def split_data(df):
    """Split into train/test. Returns X_train, y_train, X_test, y_test."""
    # Use only the first 53 for train, next 50 for test
    train_df = df.iloc[:53]
    test_df = df.iloc[53:103]
    feature_cols = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr.', 'Fine Aggr.']
    target_col = 'SLUMP(cm)'
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    return X_train, y_train, X_test, y_test