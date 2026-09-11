"""Dependency-light binary classification metrics."""

from __future__ import annotations

import numpy as np


def validate_predictions(labels: np.ndarray, predictions: np.ndarray) -> None:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.ndim != 1 or predictions.shape != labels.shape or labels.size == 0:
        raise ValueError("labels and predictions must be non-empty matching 1D arrays")
    if not np.isfinite(labels).all() or not np.isfinite(predictions).all():
        raise ValueError("labels and predictions must be finite")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    if not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("predictions must be binary")


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    validate_predictions(labels, predictions)
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        raise ValueError("labels must contain both classes")
    recall_positive = float((predictions[positives] == 1).mean())
    recall_negative = float((predictions[negatives] == 0).mean())
    return {
        "accuracy": round(float((labels == predictions).mean()), 4),
        "balanced_accuracy": round((recall_positive + recall_negative) / 2, 4),
        "positive_recall": round(recall_positive, 4),
        "negative_recall": round(recall_negative, 4),
    }
