import numpy as np
import pytest

from src.metrics import classification_metrics, validate_predictions


def test_classification_metrics_match_known_values():
    labels = np.array([1, 1, 0, 0, 0, 0])
    predictions = np.array([1, 0, 0, 0, 1, 0])
    assert classification_metrics(labels, predictions) == {
        "accuracy": 0.6667,
        "balanced_accuracy": 0.625,
        "positive_recall": 0.5,
        "negative_recall": 0.75,
    }


@pytest.mark.parametrize(
    ("labels", "predictions"),
    [
        ([], []),
        ([0, 1], [0]),
        ([[0, 1]], [[0, 1]]),
        ([0, 1], [0, 2]),
        ([0, 1], [0, np.nan]),
        ([0, 3], [0, 1]),
    ],
)
def test_rejects_invalid_prediction_arrays(labels, predictions):
    with pytest.raises(ValueError):
        validate_predictions(np.asarray(labels), np.asarray(predictions))


def test_metric_requires_both_label_classes():
    with pytest.raises(ValueError):
        classification_metrics(np.array([0, 0]), np.array([0, 0]))
