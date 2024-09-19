import torch
import torch.nn as nn
from torch.optim.adam import Adam
from typing import Self


class MetallicDL:
    @classmethod
    def load(cls, path) -> Self:
        self = cls()
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
        return self

    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def predict(self, x):
        return self.model(x)

    def train(self, x, y, epochs=1000, lr=0.01):
        criterion = nn.BCELoss()
        optimizer = Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')

        return self.model

    def evaluate(self, x, y):
        y_pred = self.model(x)
        return y_pred, y



