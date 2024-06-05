import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# This is only used in final_model.py at the root level
def process_data(filename):
    df = pd.read_csv(filename)

    substrings_to_remove = ['year', 'month', 'number', 'id', 'timestamp', 'index', 'text', 'period', 'counter']
    df.drop(columns=[col for col in df.columns if any(substring in col.lower() for substring in substrings_to_remove)], inplace=True, errors='ignore')

            # remove 'b' and ' ' in original dataset
            # df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    df = df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

            # missing values
    if df.isnull().values.any():
        print(f"{df} has missing values")
        for column in df.columns:
            if df[column].dtype == 'object':
                        # mode for categorical variables
                most_common = df[column].mode()[0]
                df[column].fillna(most_common, inplace=True)
            else:
                        # mean for numeric variables
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace=True)
                df[column] = df[column].round(1)

            # If target column is not the last column, move to the last
    class_column = None
    for col in df.columns:
        if 'class' in col.lower() or col == 'Home/Away' in col:
            class_column = col
            break
    if class_column and df.columns[-1] != class_column:
        class_col = df[class_column]
        df = df.drop(columns=[class_column])
        df['cls'] = class_col
    else:
        df.rename(columns={df.columns[-1]: 'cls'}, inplace=True)
            

    # Assign numeric labels based on class frequency
    class_counts = df['cls'].value_counts(ascending=False)
    class_mapping = {cls: i for i, cls in enumerate(class_counts.index)}
    df['cls'] = df['cls'].map(class_mapping)

            
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    target = df.iloc[:, -1].values  # Convert the last column to Numpy array
    data = df.iloc[:, :-1].values    # Convert all but the last column to Numpy array

    data = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').fillna(np.nan).values
    # target = pd.to_numeric(target, errors='coerce').fillna(np.nan).astype(int).values
    return data, target
    # return df