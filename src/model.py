"""PyTorch model used by the reference approach."""

from __future__ import annotations

from torch import nn


class TabularClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 12) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)
