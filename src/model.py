import os
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.optim.adam import Adam
from create_metafeatures import calculate_metafeatures
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

input_size = 32

class MetallicDL:
    @classmethod
    def load(cls, path):
        self = cls()
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
        return self

    def __init__(self, **kwargs):
        self.test = kwargs.get('test', False)
        self.model = nn.Sequential(
            # Functional Network
            nn.Linear(input_size, 32, bias=True),
            nn.Sigmoid(),
            nn.Linear(32, 1, bias=True),
            nn.Sigmoid(),
            # Paper Network
            # nn.Linear(input_size, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.Linear(16, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def predict(self, x):
        return self.model(x)

    def train(self, x, y, epochs=50, lr=0.01):
        criterion = nn.BCELoss()
        optimizer = Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.model(x).flatten()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            # if epoch % 1 == 0:
                # print(f'Epoch: {epoch}, Loss: {loss.item()}')

        return self.model

    def evaluate(self, x, y):
        y_pred = self.model(x)
        return y_pred, y



if __name__ == '__main__':
    # Parameters
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
    else:
        print ("MPS device not found.")
    
    metallic_dir = Path(__file__).parent.parent.resolve()
    details_dir = metallic_dir / 'src/model_details'
    os.makedirs(details_dir, exist_ok=True)
    metric = "accuracy"
    learner = "rf"

    # Train data
    data = pd.read_csv(metallic_dir / 'metafeatures.csv')
    # data = pd.read_csv('merged_metafeatures.csv')
    data = data.dropna()
    data = data.drop('dataset', axis=1)
    data = data.drop('ideal_hyperparameters', axis=1)
    data = data[data["learner"] == learner]
    data = data.drop("learner", axis=1) # type: ignore


    # One hot encode the categorical variables
    resamplers = data['resampler'].unique() # type: ignore
    encoded_columns = ['resampler']
    encoder = OneHotEncoder()
    encoded_data = pd.DataFrame(encoder.fit_transform(data[encoded_columns]).toarray())
    encoded_data.columns = encoder.get_feature_names_out(encoded_columns)
    data = data.drop(encoded_columns, axis=1)
    data = pd.concat([data, encoded_data], axis=1)
    data = data.dropna()
    print(data)

    # Cleanup columns
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc", "balanced_accuracy", "geometric_mean", "cwa", "roc_auc", "pr_auc"]

    y = torch.tensor(data[metric].to_numpy(), dtype=torch.float32)
    data[metric].to_csv(f'{details_dir}/y.csv', index=False)
    X = data.drop(metrics, axis=1)
    X.to_csv(f'{details_dir}/X.csv', index=False)

    # Pad to 64 rows
    X = torch.tensor(X.to_numpy(np.float32), dtype=torch.float32)
    X = torch.nn.functional.pad(X, (0, input_size - X.shape[1]))
    model = MetallicDL(test=True)

    print(X.shape)
    print(y.shape)
    assert model.predict(X).shape[0] == y.shape[0]

    model.train(X, y, epochs=100, lr=0.01)
    model.save(details_dir / 'model.pth')


    # Load test data
    X = calculate_metafeatures( metallic_dir / 'data/processed_datasets/collins.csv')
    X = pd.DataFrame.from_dict(X, orient='index').T
    
    for resampler in resamplers:
        x = X.copy()
        x["resampler"] = resampler
        encoded_cols = pd.DataFrame(encoder.transform(x[encoded_columns]).toarray())
        encoded_cols.columns = encoder.get_feature_names_out(encoded_columns)
        x = x.drop(encoded_columns, axis=1)
        x = x.drop("dataset", axis=1)
        x = pd.concat([x, encoded_cols], axis=1)
        x = encoded_cols
        x = x.astype(float)
        
        x = torch.tensor(x.values, dtype=torch.float)
        x = torch.nn.functional.pad(x, (0, input_size - x.shape[1]))
        pred = model.predict(x)
        pred = pred.pow(2)
        print(f'{resampler}: {pred}')


