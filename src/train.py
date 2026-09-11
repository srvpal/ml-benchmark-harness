"""Deterministic CPU training for the reference classifier."""

from __future__ import annotations

import numpy as np
import torch

from src.model import TabularClassifier


def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 2026,
    epochs: int = 120,
) -> TabularClassifier:
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("training features and labels have incompatible shapes")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("training labels must contain both binary classes")

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = TabularClassifier(features.shape[1])
    x_train = torch.from_numpy(features.astype(np.float32, copy=False))
    y_train = torch.from_numpy(labels.astype(np.float32, copy=False))
    positive_weight = torch.tensor([(labels == 0).sum() / (labels == 1).sum()])
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025, weight_decay=0.001)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    return model


def predict(model: TabularClassifier, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(features.astype(np.float32))))
    return (probabilities.numpy() >= 0.5).astype(np.int64)
