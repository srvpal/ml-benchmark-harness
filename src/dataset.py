"""Deterministic synthetic data generation and train-fitted preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    train_features: np.ndarray
    train_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    imputation_values: np.ndarray


def preprocess_split(
    train_features: np.ndarray, test_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median-impute and standardize using statistics from the training split only."""
    train = np.asarray(train_features, dtype=np.float64)
    test = np.asarray(test_features, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("train and test features must be 2D arrays with matching columns")
    if train.shape[0] == 0 or np.isnan(train).all(axis=0).any():
        raise ValueError("every training feature must contain an observed value")

    medians = np.nanmedian(train, axis=0)
    train_filled = np.where(np.isnan(train), medians, train)
    test_filled = np.where(np.isnan(test), medians, test)
    means = train_filled.mean(axis=0)
    scales = train_filled.std(axis=0)
    scales[scales == 0] = 1.0
    return (
        ((train_filled - means) / scales).astype(np.float32),
        ((test_filled - means) / scales).astype(np.float32),
        medians.astype(np.float32),
    )


def make_dataset(
    seed: int = 2026,
    n_samples: int = 800,
    n_features: int = 8,
    test_fraction: float = 0.25,
    missing_rate: float = 0.08,
) -> DatasetSplit:
    """Generate, split, and preprocess a deterministic binary classification dataset."""
    if n_samples < 20 or n_features < 5:
        raise ValueError("n_samples must be at least 20 and n_features at least 5")
    if not 0 < test_fraction < 1 or not 0 <= missing_rate < 0.5:
        raise ValueError("invalid test_fraction or missing_rate")

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, n_features))
    label_noise = rng.normal(scale=0.9, size=n_samples)
    logits = (
        1.8 * features[:, 0]
        - 1.4 * features[:, 1]
        + 1.0 * features[:, 2]
        + 0.7 * features[:, 0] * features[:, 2]
        + label_noise
        - 1.25
    )
    labels = (logits > 0).astype(np.int64)

    missing = rng.random(features.shape) < missing_rate
    features[missing] = np.nan

    indices = rng.permutation(n_samples)
    test_size = int(round(n_samples * test_fraction))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    train_features, test_features, medians = preprocess_split(
        features[train_indices], features[test_indices]
    )

    return DatasetSplit(
        train_features=train_features,
        train_labels=labels[train_indices],
        test_features=test_features,
        test_labels=labels[test_indices],
        train_indices=train_indices,
        test_indices=test_indices,
        imputation_values=medians,
    )
