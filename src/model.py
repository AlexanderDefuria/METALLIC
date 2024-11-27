import os
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.optim.adam import Adam
from create_metafeatures import calculate_metafeatures
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, OneHotEncoder
from pathlib import Path


class MetallicDL:
    @classmethod
    def load(cls, path):
        self = cls()
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
        return self

    def __init__(self, **kwargs):
        self.test = kwargs.get('test', False)
        self.input_size = kwargs.get('input_size', 58)
        self.verbose = kwargs.get('verbose', False)
        self.dropout = 0.1
        self.model = nn.Sequential(
            # Functional Network
            nn.Linear(self.input_size, 512),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.Dropout(self.dropout),
            nn.Sigmoid(),
            nn.Linear(256, 64),
            nn.Dropout(self.dropout),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Dropout(self.dropout),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
            # Paper Network
            # nn.Linear(self.input_size, 64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.Linear(16, 1),
            # nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def predict(self, x):
        # Remove dropout layers
        return self.model.eval()(x).flatten()

    def train(self, x, y, epochs=400, lr=0.001):
        criterion = nn.L1Loss()
        optimizer = Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.model(x).flatten()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0 and self.verbose:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
                print(x.shape)

        return self


def preprocess(data: pd.DataFrame, selected_target: str, encoder = None, scaler = None):
    data = data.dropna()
    data.drop(columns=[ 'ideal_hyperparameters', 'dataset'], inplace=True)
    # One hot encode the categorical data
    columns = ['learner', 'resampler']
    possible_targets = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'pr_auc', 'balanced_accuracy', 'geometric_mean']
    possible_targets.remove(selected_target)

    data = data.drop(columns=possible_targets)

    unnormalized_columns = columns + [selected_target]
    if scaler is None:
        scaler = MinMaxScaler(clip=True)
        scaler.fit(data.drop(columns=unnormalized_columns))
    data[data.drop(columns=unnormalized_columns).columns] = scaler.transform(data.drop(columns=unnormalized_columns))


    if encoder is None:
        encoder = OneHotEncoder()
        encoder.fit(data[columns])
    encoded = encoder.transform(data[columns]).toarray() # type: ignore
    data = data.drop(columns=columns)
    orginal_columns = data.columns
    data = pd.concat([data, pd.DataFrame(encoded)], axis=1, ignore_index=True)
    data.columns = list(orginal_columns) +  list(encoder.get_feature_names_out(columns))
    return data, encoder, scaler


if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print ("MPS/CUDA device not found.")
        device = torch.device("cpu")
    
    total = []
    for selected_target in ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy', 'geometric_mean']:
        for i in range(0, 100):
            metallic_dir = Path(__file__).parent.parent.resolve()
            data = pd.read_csv(metallic_dir / 'metafeatures.csv')
            data = data.reset_index(drop=True)
            data, encoder, scaler = preprocess(data, selected_target)
        
            test = data.sample(frac=0.10)
            train = data.drop(index=list(test.index))
            train_x = torch.tensor(train.drop(columns=[selected_target]).values)
            train_y = torch.tensor(train[selected_target].values)
            test_x = torch.tensor(test.drop(columns=[selected_target]).values)
            test_y = torch.tensor(test[selected_target].values)


            train_x = train_x.type(torch.float32).to(device)
            train_y = train_y.type(torch.float32).to(device)
            test_x = test_x.type(torch.float32).to(device)
            test_y = test_y.type(torch.float32).to(device)
        
            model = MetallicDL(input_size=train_x.shape[1], verbose=False)
            model.train(train_x, train_y)
        
            # Test using mean squared error
            pred = model.predict(test_x)
            loss = nn.L1Loss()
            total.append([selected_target, loss(pred, test_y).item()])
    
    df = pd.DataFrame(total, columns=['Target', 'Error'])
    print(f'Mean Average Error: {np.mean(df["Error"])}')
    print(f'Standard Deviation: {np.std(df["Error"])}')




    
