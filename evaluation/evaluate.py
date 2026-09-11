"""Run baseline and reference approaches against the fixed synthetic benchmark."""

from __future__ import annotations

import argparse
import json

from baselines.baseline import majority_predictions
from reference.reference import reference_predictions
from src.dataset import make_dataset
from src.metrics import classification_metrics, validate_predictions


MINIMUM_REFERENCE_BALANCED_ACCURACY = 0.75


def run_evaluation(seed: int = 2026) -> dict:
    dataset = make_dataset(seed=seed)
    baseline = majority_predictions(dataset.train_labels, len(dataset.test_labels))
    reference = reference_predictions(
        dataset.train_features,
        dataset.train_labels,
        dataset.test_features,
        seed=seed,
    )
    validate_predictions(dataset.test_labels, baseline)
    validate_predictions(dataset.test_labels, reference)
    baseline_metrics = classification_metrics(dataset.test_labels, baseline)
    reference_metrics = classification_metrics(dataset.test_labels, reference)
    criteria = {
        "reference_above_baseline": (
            reference_metrics["balanced_accuracy"]
            > baseline_metrics["balanced_accuracy"]
        ),
        "reference_meets_minimum": (
            reference_metrics["balanced_accuracy"]
            >= MINIMUM_REFERENCE_BALANCED_ACCURACY
        ),
    }
    return {
        "benchmark": "synthetic-tabular-classification",
        "seed": seed,
        "dataset": {
            "train_samples": len(dataset.train_labels),
            "test_samples": len(dataset.test_labels),
            "features": dataset.train_features.shape[1],
            "train_positive_rate": round(float(dataset.train_labels.mean()), 4),
            "test_positive_rate": round(float(dataset.test_labels.mean()), 4),
        },
        "primary_metric": "balanced_accuracy",
        "baseline": baseline_metrics,
        "reference": reference_metrics,
        "success_criteria": criteria,
        "benchmark_passed": all(criteria.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic ML benchmark")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = run_evaluation(args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["benchmark_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
