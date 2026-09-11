# ML Benchmark Harness

This repository contains a deterministic synthetic tabular classification
benchmark. It generates data with NumPy, trains a small PyTorch model on CPU, and
emits a structured JSON comparison with a majority-class baseline.

## Problem

Predict a binary label from eight numeric features. The benchmark is designed to
exercise data generation, train/test separation, missing-value preprocessing,
model training, prediction validation, and metric calculation in one small run.

## Dataset

`src/dataset.py` generates 800 synthetic rows from a fixed NumPy random seed.
Three features carry most of the signal, one interaction contributes additional
signal, and the remaining features are noise. Label noise makes the classes
overlap. A negative intercept produces class imbalance, and 8% of feature values
are replaced with missing values.

The default split contains 600 training rows and 200 test rows. Rows are assigned
to the split before preprocessing. Median imputation, centering, and scaling are
fitted on the training rows and then applied to the test rows.

## Rules

- No external data is downloaded.
- Random seeds control NumPy generation, splitting, and PyTorch initialization.
- Test labels and test feature statistics are not used during training or preprocessing.
- Predictions must be a one-dimensional binary array matching the test labels.
- Balanced accuracy is the primary metric because the classes are imbalanced.
- A run passes when the reference balanced accuracy is at least 0.75 and exceeds the baseline.

## Baseline

`baselines/baseline.py` predicts the majority class observed in the training
labels. On the default seed-2026 data, its measured test metrics are:

```json
{"accuracy": 0.725, "balanced_accuracy": 0.5, "negative_recall": 1.0, "positive_recall": 0.0}
```

## Reference Approach

`reference/reference.py` trains a two-layer PyTorch classifier. Training uses
binary cross-entropy with a positive-class weight computed from training labels,
Adam optimization, deterministic operations, and CPU tensors. On the same test
split, the measured metrics are:

```json
{"accuracy": 0.865, "balanced_accuracy": 0.8392, "negative_recall": 0.8966, "positive_recall": 0.7818}
```

These values describe the committed synthetic benchmark and fixed implementation;
they are not claims about other datasets or models.

## Evaluation

Run:

```bash
python -m evaluation.evaluate
```

The evaluator runs both approaches, validates their predictions, calculates
accuracy, balanced accuracy, and per-class recall, checks the success criteria,
prints sorted JSON, and exits nonzero if the benchmark criteria fail.

## Reproducibility

The default seed is 2026. Direct dependencies are pinned, preprocessing uses only
training data, PyTorch deterministic algorithms are enabled, and GitHub Actions
runs the tests and benchmark with Python 3.12 on CPU.

A Dockerfile is omitted because the repository has only three Python dependencies
and the CI workflow already defines a clean runnable environment. PyTorch also
makes a container substantially larger than the source benchmark.

## Run Locally

```bash
python -m pip install -r requirements.txt
python -m evaluation.evaluate
```

Use another deterministic dataset and training seed with `--seed`:

```bash
python -m evaluation.evaluate --seed 7
```

## Tests

```bash
python -m pytest
```

The tests cover deterministic generation, split isolation, training-only
preprocessing, malformed configurations, metric arithmetic, invalid predictions,
baseline and reference output shapes, deterministic evaluation JSON, and the
benchmark success criteria.

## Repository Structure

```text
src/
  dataset.py
  model.py
  train.py
  metrics.py
baselines/
  baseline.py
reference/
  reference.py
evaluation/
  evaluate.py
tests/
  test_dataset.py
  test_metrics.py
  test_evaluation.py
.github/workflows/tests.yml
requirements.txt
```

## Limitations

The data is synthetic and the fixed success threshold is specific to this task.
The benchmark does not measure calibration, uncertainty, fairness, distribution
shift, or performance on real-world datasets. It compares one simple baseline
with one small reference model rather than surveying model families.
