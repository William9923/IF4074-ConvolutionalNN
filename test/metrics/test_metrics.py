import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from src.metrics import Metrics


@pytest.mark.parametrize(
    "name, test, prediction, expected_output",
    [
        (
            "Test Accuracy - 100%",
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 1]),
            1,
        ),
        (
            "Test Accuracy - 50%",
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 0, 0]),
            0.5,
        ),
        (
            "Test Accuracy - 0%",
            np.array([1, 0, 1, 0]),
            np.array([0, 1, 0, 1]),
            0,
        ),
    ],
)
def test_Metrics_accuracy(name, test, prediction, expected_output):
    score = Metrics.accuracy(test, prediction)
    try:
        assert_array_almost_equal(expected_output, score, decimal=5)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {score}"


@pytest.mark.parametrize(
    "name, test, prediction, expected_output",
    [
        (
            "Test Precision - 100%",
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 1]),
            1,
        ),
        (
            "Test Precision - 50%",
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 0, 0]),
            0.5,
        ),
        (
            "Test Precision - Random",
            np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0]),
            np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
            0.5,
        ),
    ],
)
def test_Metrics_precision(name, test, prediction, expected_output):
    score = Metrics.precision(test, prediction)
    print(score)
    try:
        assert_array_almost_equal(expected_output, score, decimal=5)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {score}"


@pytest.mark.parametrize(
    "name, test, prediction, expected_output",
    [
        (
            "Test Recall - 100%",
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 1]),
            1,
        ),
        (
            "Test Recall - 50%",
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 0, 0]),
            0.5,
        ),
        (
            "Test Recall - Random",
            np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0]),
            np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
            0.6,
        ),
    ],
)
def test_Metrics_recall(name, test, prediction, expected_output):
    score = Metrics.recall(test, prediction)
    try:
        assert_array_almost_equal(expected_output, score, decimal=5)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {score}"


@pytest.mark.parametrize(
    "name, test, prediction, expected_output",
    [
        (
            "Test F1 - 100%",
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 1]),
            1,
        ),
        (
            "Test F1 - 50%",
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 0, 0]),
            0.5,
        ),
        (
            "Test F1 - Random",
            np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0]),
            np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1]),
            0.545454,
        ),
    ],
)
def test_Metrics_f1(name, test, prediction, expected_output):
    score = Metrics.f1(test, prediction)
    try:
        assert_array_almost_equal(expected_output, score, decimal=5)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {score}"


@pytest.mark.parametrize(
    "name, test, prediction, expected_output",
    [
        (
            "Test Confusion Matrix - Type 1",
            np.array([1, 1, 1, 1]),
            np.array([1, 1, 1, 1]),
            np.array(
                [
                    [4, 0],
                    [0, 0],
                ]
            ),
        ),
        (
            "Test Confusion Matrix - Type 2",
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 0, 0]),
            np.array(
                [
                    [1, 1],
                    [1, 1],
                ]
            ),
        ),
        (
            "Test Confusion Matrix - Type 3",
            np.array([1, 0, 1, 0]),
            np.array([0, 0, 0, 0]),
            np.array(
                [
                    [0, 0],
                    [2, 2],
                ]
            ),
        ),
    ],
)
def test_Metrics_confusion_matrix(name, test, prediction, expected_output):
    score = Metrics.confusion_matrix(test, prediction)
    try:
        assert_array_equal(expected_output, score)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {score}"
