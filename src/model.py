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
        self.model = nn.Sequential(
            # Functional Network
            nn.Linear(self.input_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
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
        ).to(mps_device)

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def predict(self, x):
        return self.model(x)

    def train(self, x, y, epochs=500, lr=0.001):
        criterion = nn.L1Loss()
        optimizer = Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.model(x).flatten()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
                print(x.shape)

        return self


def preprocess(data: pd.DataFrame, encoder = None, scaler = None):
    data = data.dropna()
    data.drop(columns=[ 'ideal_hyperparameters', 'dataset'], inplace=True)
    # One hot encode the categorical data
    columns = ['learner', 'resampler']
    possible_targets = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'pr_auc', 'balanced_accuracy', 'geometric_mean']
    selected_target = 'accuracy'
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
        mps_device = torch.device("mps")
    else:
        print ("MPS device not found.")
        raise SystemExit
    
    metallic_dir = Path(__file__).parent.parent.resolve()
    data = pd.read_csv(metallic_dir / 'metafeatures.csv')
    data = data[data['dataset'] != 'collins']
    data = data.reset_index(drop=True)
    data, encoder, scaler = preprocess(data)
    selected_target = 'accuracy'

    test = data.sample(frac=0.01)
    train = data.drop(index=list(test.index))
    train_x = torch.tensor(train.drop(columns=[selected_target]).values).type(torch.float32)
    train_y = torch.tensor(train[selected_target].values).type(torch.float32)
    test_x = torch.tensor(test.drop(columns=[selected_target]).values).type(torch.float32)
    test_y = torch.tensor(test[selected_target].values).type(torch.float32)
    train_x = train_x.to(mps_device)
    train_y = train_y.to(mps_device)
    test_x = test_x.to(mps_device)
    test_y = test_y.to(mps_device)

    model = MetallicDL(input_size=train_x.shape[1])
    model.train(train_x, train_y)

    # Test using mean squared error
    pred = model.predict(test_x)
    loss = nn.MSELoss()
    print(f'Mean Average Error: {loss(pred, test_y).item()}')


    vdf = pd.read_csv(metallic_dir / 'metafeatures.csv')
    # vdf = vdf[vdf['dataset'] == 'wine']
    vdf = vdf[vdf['dataset'] == 'collins']
    vdf = vdf.reset_index(drop=True)
    vdf, _, _ = preprocess(vdf, encoder, scaler)

    one_hot_columns = [col for col in vdf.columns if 'learner' in col or 'resampler' in col]
    text_labels = encoder.inverse_transform(vdf[one_hot_columns])

    pred = model.predict(torch.tensor(vdf.drop(columns=[selected_target]).values).type(torch.float32).to(mps_device))
    true = torch.tensor(vdf[selected_target].values).type(torch.float32).to(mps_device)
    
    for pred_i, true_i, labels in zip(pred.flatten().tolist(), true.tolist(), text_labels):
        print(f'Predicted: {pred_i:.3f}, True: {true_i:.3f} for {labels[0]} with {labels[1]}')

    print(f"Mean Average Error for the validation set: {loss(pred, true).item()}")




    


