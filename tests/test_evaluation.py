import json

import numpy as np
import pytest

from baselines.baseline import majority_predictions
from evaluation.evaluate import run_evaluation
from reference.reference import reference_predictions
from src.dataset import make_dataset
from src.metrics import validate_predictions


def test_baseline_and_reference_produce_valid_prediction_shapes():
    data = make_dataset()
    baseline = majority_predictions(data.train_labels, len(data.test_labels))
    reference = reference_predictions(
        data.train_features, data.train_labels, data.test_features
    )
    validate_predictions(data.test_labels, baseline)
    validate_predictions(data.test_labels, reference)
    assert baseline.shape == data.test_labels.shape
    assert reference.shape == data.test_labels.shape


def test_evaluation_is_deterministic_and_json_serializable():
    first = run_evaluation()
    second = run_evaluation()
    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first


def test_benchmark_success_criteria_are_satisfied():
    result = run_evaluation()
    assert result["primary_metric"] == "balanced_accuracy"
    assert result["success_criteria"] == {
        "reference_above_baseline": True,
        "reference_meets_minimum": True,
    }
    assert result["benchmark_passed"] is True


@pytest.mark.parametrize(
    ("labels", "size"),
    [([], 1), ([0, 2], 2), ([0, 1], 0), ([0, 1], True)],
)
def test_baseline_rejects_invalid_inputs(labels, size):
    with pytest.raises(ValueError):
        majority_predictions(np.asarray(labels), size)
