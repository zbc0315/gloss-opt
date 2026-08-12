import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


class _BaseNNRegressor(BaseEstimator, RegressorMixin):
    """Base sklearn-compatible wrapper for PyTorch regression models."""

    def __init__(self, hidden_sizes=(64, 32), lr=0.01, epochs=200, batch_size=32,
                 random_state=42):
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _build_network(self, input_dim):
        raise NotImplementedError

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        self.scaler_X_ = StandardScaler().fit(X)
        self.scaler_y_ = StandardScaler().fit(y.reshape(-1, 1))
        X_s = torch.FloatTensor(self.scaler_X_.transform(X))
        y_s = torch.FloatTensor(self.scaler_y_.transform(y.reshape(-1, 1)).ravel())

        self.net_ = self._build_network(X.shape[1])
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.net_.train()
        n = len(X_s)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                pred = self.net_(X_s[idx]).squeeze()
                loss = loss_fn(pred, y_s[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.net_.eval()
        X_s = torch.FloatTensor(self.scaler_X_.transform(X))
        with torch.no_grad():
            pred_s = self.net_(X_s).squeeze().numpy()
        return self.scaler_y_.inverse_transform(pred_s.reshape(-1, 1)).ravel()


class NNRegressor(_BaseNNRegressor):
    """Plain feedforward neural network."""

    def _build_network(self, input_dim):
        layers = []
        prev = input_dim
        for h in self.hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)


class _ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.fc2(self.relu(self.fc1(x))))


class ResNetRegressor(_BaseNNRegressor):
    """ResNet-style network with skip connections."""

    def __init__(self, hidden_size=64, n_blocks=3, **kwargs):
        super().__init__(hidden_sizes=(hidden_size,), **kwargs)
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks

    def _build_network(self, input_dim):
        layers = [nn.Linear(input_dim, self.hidden_size), nn.ReLU()]
        for _ in range(self.n_blocks):
            layers.append(_ResBlock(self.hidden_size))
        layers.append(nn.Linear(self.hidden_size, 1))
        return nn.Sequential(*layers)


class _HighwayBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transform = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        t = torch.sigmoid(self.gate(x))
        h = self.relu(self.transform(x))
        return t * h + (1 - t) * x


class HighwayNNRegressor(_BaseNNRegressor):
    """Highway network with gated skip connections."""

    def __init__(self, hidden_size=64, n_blocks=3, **kwargs):
        super().__init__(hidden_sizes=(hidden_size,), **kwargs)
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks

    def _build_network(self, input_dim):
        layers = [nn.Linear(input_dim, self.hidden_size), nn.ReLU()]
        for _ in range(self.n_blocks):
            layers.append(_HighwayBlock(self.hidden_size))
        layers.append(nn.Linear(self.hidden_size, 1))
        return nn.Sequential(*layers)


def get_nn_model_configs(input_dim):
    """Return list of NN model configs for auto-selection."""
    return [
        {
            "name": "NN",
            "estimator": NNRegressor(hidden_sizes=(64, 32), epochs=200),
            "param_grid": {
                "lr": [0.001, 0.01],
                "epochs": [100, 200],
            },
        },
        {
            "name": "ResNet",
            "estimator": ResNetRegressor(hidden_size=64, n_blocks=3, epochs=200),
            "param_grid": {
                "lr": [0.001, 0.01],
                "hidden_size": [32, 64],
                "n_blocks": [2, 3],
            },
        },
        {
            "name": "HighwayNN",
            "estimator": HighwayNNRegressor(hidden_size=64, n_blocks=3, epochs=200),
            "param_grid": {
                "lr": [0.001, 0.01],
                "hidden_size": [32, 64],
                "n_blocks": [2, 3],
            },
        },
    ]
