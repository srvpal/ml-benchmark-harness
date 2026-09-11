"""Train and run the deterministic PyTorch reference classifier."""

from __future__ import annotations

import numpy as np

from src.train import predict, train_classifier


def reference_predictions(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    seed: int = 2026,
) -> np.ndarray:
    model = train_classifier(train_features, train_labels, seed=seed)
    return predict(model, test_features)
