import numpy as np
import pytest

from src.dataset import make_dataset, preprocess_split


def test_dataset_generation_is_deterministic():
    first = make_dataset()
    second = make_dataset()
    for field in first.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


def test_split_shapes_classes_and_missing_value_preprocessing():
    data = make_dataset()
    assert data.train_features.shape == (600, 8)
    assert data.test_features.shape == (200, 8)
    assert data.train_labels.shape == (600,)
    assert data.test_labels.shape == (200,)
    assert not np.isnan(data.train_features).any()
    assert not np.isnan(data.test_features).any()
    assert set(data.train_labels) == {0, 1}
    assert set(data.test_labels) == {0, 1}
    assert data.train_labels.mean() < 0.5


def test_train_and_test_indices_are_disjoint_and_complete():
    data = make_dataset()
    assert set(data.train_indices).isdisjoint(data.test_indices)
    assert set(np.concatenate([data.train_indices, data.test_indices])) == set(range(800))


def test_preprocessing_statistics_come_only_from_training_data():
    train = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 8.0]])
    ordinary_test = np.array([[np.nan, 6.0]])
    extreme_test = np.array([[np.nan, 60_000.0]])

    train_one, _, medians_one = preprocess_split(train, ordinary_test)
    train_two, _, medians_two = preprocess_split(train, extreme_test)

    np.testing.assert_array_equal(train_one, train_two)
    np.testing.assert_array_equal(medians_one, medians_two)
    np.testing.assert_array_equal(medians_one, np.array([3.0, 6.0], dtype=np.float32))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 10},
        {"n_features": 4},
        {"test_fraction": 0},
        {"test_fraction": 1},
        {"missing_rate": -0.1},
        {"missing_rate": 0.5},
    ],
)
def test_rejects_invalid_dataset_configuration(kwargs):
    with pytest.raises(ValueError):
        make_dataset(**kwargs)
