import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import math
from math import sqrt


def preprocess(filename):
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
    class_column = None
    for col in df.columns:
        if "class" in col.lower() in col:
            class_column = col
            break
    df = df[[col for col in df.columns if col != class_column] + [class_column]]    

    # Assign numeric labels based on class frequency
    class_counts = df["cls"].value_counts(ascending=False)
    class_mapping = {cls: i for i, cls in enumerate(class_counts.index)}
    df["cls"] = df["cls"].map(class_mapping)

    # Encode categorical variables using the label encoder
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == "object":
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    target = df.iloc[:, -1].values  # Convert the last column to Numpy array
    data = df.iloc[:, :-1].values  # Convert all but the last column to Numpy array

    data = pd.DataFrame(data).apply(pd.to_numeric, errors="coerce").fillna(np.nan).values
    return data, target



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