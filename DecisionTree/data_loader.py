import zipfile
from typing import List, Dict


def load_car_data(zip_path: str, split: str = "train") -> List[Dict[str, str]]:
    """
    Loads car evaluation data from a zip file.
    Args:
        zip_path: Path to the car-4.zip file.
        split: 'train' or 'test' to select the file inside the zip.
    Returns:
        List of dictionaries, each representing a row with attribute names as keys.
    """
    # Map split to file name
    file_map = {"train": "train.csv", "test": "test.csv"}
    if split not in file_map:
        raise ValueError("split must be 'train' or 'test'")
    csv_file = file_map[split]

    with zipfile.ZipFile(zip_path) as z:
        # Read column names from data-desc.txt
        desc = z.read("data-desc.txt").decode("utf-8")
        # Find the line with column names
        columns = None
        for line in desc.splitlines():
            if line.strip().startswith("buying,"):
                columns = [col.strip() for col in line.strip().split(",")]
                break
        if columns is None:
            raise RuntimeError("Could not find column names in data-desc.txt")

        # Read the CSV file
        with z.open(csv_file) as f:
            data = []
            for row in f:
                row = row.decode("utf-8").strip()
                if not row:
                    continue
                values = [v.strip() for v in row.split(",")]
                if len(values) != len(columns):
                    raise ValueError(f"Row has {len(values)} values, expected {len(columns)}: {row}")
                data.append(dict(zip(columns, values)))
    return data


def load_bank_data(zip_path: str, split: str = "train") -> List[Dict[str, str]]:
    """
    Loads bank marketing data from a zip file.
    Args:
        zip_path: Path to the bank-4.zip file.
        split: 'train' or 'test' to select the file inside the zip.
    Returns:
        List of dictionaries, each representing a row with attribute names as keys.
        Numeric columns are cast to float (or int where appropriate).
    """
    # Map split to file name
    file_map = {"train": "train.csv", "test": "test.csv"}
    if split not in file_map:
        raise ValueError("split must be 'train' or 'test'")
    csv_file = file_map[split]

    # Column names and types (from data-desc.txt and UCI info)
    columns = [
        "age", "job", "marital", "education", "default", "balance", "housing", "loan",
        "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"
    ]
    numeric_cols = {"age", "balance", "day", "duration", "campaign", "pdays", "previous"}

    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_file) as f:
            data = []
            for row in f:
                row = row.decode("utf-8").strip()
                if not row:
                    continue
                values = [v.strip() for v in row.split(",")]
                if len(values) != len(columns):
                    raise ValueError(f"Row has {len(values)} values, expected {len(columns)}: {row}")
                row_dict = {}
                for col, val in zip(columns, values):
                    if col in numeric_cols:
                        # Try int, fallback to float
                        try:
                            row_dict[col] = int(val)
                        except ValueError:
                            row_dict[col] = float(val)
                    else:
                        row_dict[col] = val
                data.append(row_dict)
    return data 