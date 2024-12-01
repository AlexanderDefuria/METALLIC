import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import joblib
from pathlib import Path
from torch.optim.adam import Adam
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def preprocess(data: pd.DataFrame, selected_target: str, encoder = None, scaler = None):
    """
    Preprocess a dataset that already has the metafeatures calculated. The encoder and scaler can be passed back in to preprocess the test dataset using the same scalings and encodings. This helps prevent test set leakage.

    :param data: The dataset to preprocess
    :param selected_target: The target to predict
    :param encoder: The encoder to use
    :param scaler: The scaler to use
    :return: The preprocessed dataset, the encoder used, and the scaler used
    """
    data = data.dropna()
    data.drop(columns=[ 'ideal_hyperparameters', 'dataset'], inplace=True)
    # One hot encode the categorical data
    columns = ['learner', 'resampler']
    possible_targets = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'pr_auc', 'balanced_accuracy', 'geometric_mean', 'cwa']
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


class MetallicDL:
    @classmethod
    def load(cls, path):
        self = cls()
        self.model.load_state_dict(torch.load(path.with_suffix(".pth"), weights_only=True))
        self.model.eval()
        return self

    def __init__(self, **kwargs):
        self.test = kwargs.get("test", False)
        self.input_size = kwargs.get("input_size", 58)
        self.verbose = kwargs.get("verbose", False)
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
            nn.Sigmoid(),
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
        torch.save(self.model.state_dict(), path.with_suffix(".pth"))

    def predict(self, x):
        # Remove dropout layers
        x = self._handle_types(x)
        return self.model.eval()(x).flatten()

    def train(
        self,
        x: torch.Tensor | pd.DataFrame,
        y: torch.Tensor | pd.DataFrame | pd.Series,
        epochs=400,
        lr=0.001,
    ):
        criterion = nn.L1Loss()
        optimizer = Adam(self.model.parameters(), lr=lr)
        x, y = self._handle_types(x), self._handle_types(y)

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.model(x).flatten()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0 and self.verbose:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
                print(x.shape)

        return self

    @staticmethod
    def _handle_types(x: torch.Tensor | pd.DataFrame | pd.Series | np.ndarray):
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values).type(torch.float32).to(device)
        if isinstance(x, pd.Series):
            x = torch.tensor(x.values).type(torch.float32).to(device)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).type(torch.float32).to(device)
        return x

class MetallicXGB:
    @classmethod
    def load(cls, path):
        self = cls()
        self.model.load_model(path.with_suffix(".json"))
        return self

    def __init__(self, **kwargs):
        self.model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            n_jobs=-1,
            verbosity=0,
            objective="reg:squarederror",
        )

    def save(self, path: Path):
        self.model.save_model(path.with_suffix(".json"))

    def predict(self, x):
        return self.model.predict(x)
    def train(self, x, y):
        self.model.fit(x, y)
        return self

class MetallicKNN():
    @classmethod
    def load(cls, path):
        self = cls()
        self.model = joblib.load(path.with_suffix(".joblib"))
        return self

    def __init__(self, **kwargs):
        self.model = KNeighborsRegressor(n_neighbors=3, n_jobs=-1)
    def predict(self, x):
        return self.model.predict(x)
    def train(self, x, y):
        self.model.fit(x, y)
        return self

    def save(self, path: Path):
        joblib.dump(self.model, path.with_suffix(".joblib"))



if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    metallic_dir = Path(__file__).parent.parent.resolve()
    models_dir = metallic_dir / "src" / "saved_models"
    total = []
    for selected_target in ["accuracy"]:
        for ModelClass in [MetallicKNN, MetallicXGB, MetallicDL]:
            for i in range(0, 100):
                data = pd.read_csv(metallic_dir / "metafeatures.csv")
                data = data.reset_index(drop=True)
                data, encoder, scaler = preprocess(data, selected_target)

                test = data.sample(frac=0.10)
                train = data.drop(index=list(test.index))
                train_x = train.drop(columns=[selected_target])
                train_y = train[selected_target]
                test_x = test.drop(columns=[selected_target])
                test_y = test[selected_target]

                model = ModelClass(input_size=train_x.shape[1], verbose=False)
                model_name = model.__class__.__name__
                model.train(train_x, train_y)
                model.save(models_dir / f"{model_name}_{selected_target}")

                # Test using mean squared error
                pred = model.predict(test_x)
                test_y = MetallicDL._handle_types(test_y)
                pred = MetallicDL._handle_types(pred)
                loss = nn.L1Loss()
                total.append([selected_target, model_name, loss(pred, test_y).item()])
                break

    overall_df = pd.DataFrame(total, columns=["Target", "Model", "Error"])

    for index, df in overall_df.groupby(["Target", "Model"]):
        print(f'{index[1]}, {index[0]} MAE: {np.mean(df["Error"])}') # type: ignore
        print(f'{index[1]}, {index[0]} SD: {np.std(df["Error"])}') # type: ignore
