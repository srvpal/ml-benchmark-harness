"""Majority-class baseline fitted only on training labels."""

from __future__ import annotations

import numpy as np


def majority_predictions(train_labels: np.ndarray, n_predictions: int) -> np.ndarray:
    labels = np.asarray(train_labels)
    if labels.ndim != 1 or labels.size == 0 or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("train_labels must be a non-empty binary array")
    if not isinstance(n_predictions, int) or isinstance(n_predictions, bool) or n_predictions < 1:
        raise ValueError("n_predictions must be a positive integer")
    majority = int(labels.mean() >= 0.5)
    return np.full(n_predictions, majority, dtype=np.int64)
