from locale import normalize
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, normalize, minmax_scale
import math
from typing import List
from pathlib import Path
from tqdm import tqdm
from scipy.io import arff
import os

DATA_DIR: Path = Path(__file__).parent.parent / "data"

def get_datasets(path: Path = Path(__file__).parent.parent / "data"):
    DATA_DIR = path
    collect_arrf_datasets()
    datasets = collect_datasets(raw_datasets_dir())
    datasets = [
        dataset
        for dataset in datasets
        if dataset.stem
        not in [
            "primary-tumor",
            "movement_libras",  # Fails with KNN & SMOTEENN
        ]
    ]  # Filter datasets

    #for dataset in tqdm(datasets, desc="Preprocessing Datasets to ../data/processed_datasets"):
    for dataset in datasets:
        df_processed: pd.DataFrame | None = preprocess(dataset)
        if df_processed is not None:
            if df_processed["cls"].value_counts().min() > 5:
                df_processed.to_csv(processed_datasets_dir() / dataset.name, index=False)

    return collect_datasets(processed_datasets_dir())


def dataset_dir() -> Path:
    return DATA_DIR


def arrf_datasets_dir() -> Path:
    return DATA_DIR / "arrf_datasets"


def processed_datasets_dir() -> Path:
    path = DATA_DIR / "processed_datasets"
    if not path.exists():
        os.makedirs(path)
    return path


def raw_datasets_dir() -> Path:
    return DATA_DIR / "raw_datasets"


def collect_datasets(directory: Path) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        return []

    return list(directory.glob("*.csv"))


def collect_arrf_datasets() -> List[Path]:
    """
    Collects ARFF datasets and converts them to CSV format if they have not been converted before.
    Returns:
        List[Path]: A list of paths to the converted CSV datasets.
    """

    existing_csv_datasets = collect_datasets(raw_datasets_dir())
    existing_csv_datasets = [dataset.stem for dataset in existing_csv_datasets]
    arrf_datasets = list(arrf_datasets_dir().glob("*.arff"))
    arrf_to_process = [dataset for dataset in arrf_datasets if dataset.stem not in existing_csv_datasets]
    converted_paths = []

    for dataset in arrf_to_process:
        data, _ = arff.loadarff(dataset)
        df = pd.DataFrame(data)
        path = (raw_datasets_dir() / dataset.stem).with_suffix(".csv")
        print(f"Converting {dataset} to {path}")
        df.to_csv(path, index=False)
        converted_paths.append(path)

    return converted_paths


def preprocess(filename) -> pd.DataFrame:
    """
    Steps:
    - Read the CSV file.
    - Drop superfluous columns.
    - Format to UTF-8.
    - Handle missing values.
        - Mode for categorical variables.
        - Mean for numeric variables.
    - Move target column to the last column for ease of access
    - Assign numeric labels based on class frequency.
    - Encode categorical variables using the label encoder.
    TODO (optional): Implement one-hot encoding for categorical variables.
    TODO (optional): Implement other missing value handling techniques.
    TODO: Review missing value handling techniques.

    Args:
        filename (str): The path to the CSV file.
    Returns:
        tuple: A tuple containing the preprocessed data and the target values.
    """

    df = pd.read_csv(filename)

    substrings_to_remove = ["year", "month", "number", "id", "timestamp", "index", "text", "period", "counter"]
    columns = [col for col in df.columns if any(substring in col.lower() for substring in substrings_to_remove)]
    df.drop(columns=columns, inplace=True, errors="ignore")
    df = df.map(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    # Handle Missing Values
    if df.isnull().values.any():
        for column in df.columns:
            if df[column].dtype == "object":
                # mode for categorical variables
                most_common = df[column].mode()[0]
                df[column].fillna(most_common, inplace=True)
            else:
                # mean for numeric variables
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace=True)
                df[column] = df[column].round(1)

    # Grab the target column and move it to the last column
    possible_class_columns = ["cls", "class", "label", "target", "output", "result", "type"]
    class_column = None
    for col in possible_class_columns:
        if col in df.columns.str.lower():
            index = df.columns.str.lower().tolist().index(col)
            class_column = df.columns[index]
            break

    if class_column is None:
        # Check if the target column is the last column if it is categorical
        last_column = df.columns[-1]
        if df[last_column].dtype == "object":
            class_column = last_column
        elif df[last_column].dtype == "int":
            class_column = last_column
            # print(f"Target column {class_column} is the last column in the dataset {filename}")

    if class_column is None:
        raise ValueError(f"No class column found in the dataset {filename}")
    df = df[[col for col in df.columns if col != class_column] + [class_column]]
    df.rename(columns={class_column: "cls"}, inplace=True)

    # Assign numeric labels based on class frequency
    class_counts = df["cls"].value_counts(ascending=False)

    # If there are any classes less than 5 remove the dataset
    if class_counts.min() < 5:
        return None

    class_mapping = {cls: i for i, cls in enumerate(class_counts.index)}
    df["cls"] = df["cls"].map(class_mapping)

    # Encode categorical variables using the label encoder
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = LabelEncoder().fit_transform(df[column])

    assert df["cls"][np.isnan(df["cls"])].size == 0
    assert df["cls"].value_counts().min() > 1, f"{filename} has a class with only one instance"
    assert df["cls"].unique().max() == len(df["cls"].unique()) - 1, f"{filename} has missing classes"

    # Ensure that the cls is in ascending from 0 to n
    df["cls"] = LabelEncoder().fit_transform(df["cls"])

    return pd.DataFrame(df).apply(pd.to_numeric, errors="coerce").fillna(np.nan)


# THIS IS LEGACY CODE BUT MAY BE USEFUL FOR FUTURE REFERENCE
def handle_missing_values(X, y, data_clean):

    option = data_clean

    if option == 1:
        rows, cols = X.shape
        meansArray = []
        class_label_mapper = {}
        index = 0
        for i in np.unique(y):
            class_label_mapper[i] = index
            index += 1

        for i in range(len(np.unique(y))):
            meansArray.append([])
            for j in range(X.shape[1]):
                meansArray[i].append(np.nanmean(X[tuple(list(np.where(y == i)))][:, j]))
        for i in range(rows):
            for j in range(cols):
                if math.isnan(X[i][j]):
                    if math.isnan(meansArray[class_label_mapper[y[i]]][j]):
                        X[i][j] = 0
                    else:
                        X[i][j] = meansArray[class_label_mapper[y[i]]][j]
        return X

    elif option == 2:
        X = X[~np.isnan(X).any(axis=1)]
        return X
    elif option == 3:
        X = X[:, ~np.isnan(X).any(axis=0)]
        return X
    else:
        print("Invalid data cleaning option\n")
