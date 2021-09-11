import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
from src.loss import Loss


@pytest.mark.parametrize(
    "name, target, predicted, expected_output",
    [
        (
            "Case 1 : MSE - Regression",
            np.array([34, 37, 44, 47, 48, 48, 46, 43, 32, 27, 26, 24]),
            np.array([37, 40, 46, 44, 46, 50, 45, 44, 34, 30, 22, 23]),
            5.91667 * 1 / 2,
        ),
        (
            "Case 2 : MSE - Probability",
            np.array([0.8, 0.1, 0.5, 0.9, 0.55, 0.6]),
            np.array([0.6, 0.3, 0.6, 0.8, 0.6, 0.8]),
            0.02375 * 1 / 2,
        ),
    ],
)
def test_Loss_mse(name, target, predicted, expected_output):
    loss = Loss.mse(target, predicted)
    try:
        assert_array_almost_equal(expected_output, loss, decimal=3)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {loss}"


@pytest.mark.parametrize(
    "name, target, predicted, expected_output",
    [
        (
            "Case 1 : MSE (derivative) - Regression",
            np.array([34, 37, 44, 47, 48, 48, 46, 43, 32, 27, 26, 24]),
            np.array([37, 40, 46, 44, 46, 50, 45, 44, 34, 30, 22, 23]),
            np.array([3, 3, 2, -3, -2, 2, -1, 1, 2, 3, -4, -1]),
        ),
        (
            "Case 2 : MSE (derivative) - Probability",
            np.array([0.8, 0.1, 0.5, 0.9, 0.55, 0.6]),
            np.array([0.6, 0.3, 0.6, 0.8, 0.6, 0.8]),
            np.array([-0.2, 0.2, 0.1, -0.1, 0.05, 0.2]),
        ),
    ],
)
def test_Loss_mse_derivative(name, target, predicted, expected_output):
    loss = Loss.mse(target, predicted, deriv=True)
    try:
        assert_array_almost_equal(expected_output, loss, decimal=3)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {loss}"


@pytest.mark.parametrize(
    "name, target, predicted, expected_output",
    [
        (
            "Case 1 : Binary Cross Entropy - Good case",
            np.array([1, 0]),
            np.array([0.85, 0.25]),
            0.2251,
        ),
        (
            "Case 2 : Binary Cross Entropy - Bad case",
            np.array([1, 0]),
            np.array([0.15, 0.75]),
            1.6417,
        ),
        (
            "Case 3 : Binary Cross Entropy - Tensorflow example",
            np.array([0, 1, 0, 0]),
            np.array([0.6, 0.3, 0.2, 0.8]),
            0.988211,
        ),
    ],
)
def test_Loss_binary_cross_entropy(name, target, predicted, expected_output):
    loss = Loss.binary_cross_entropy(target, predicted, deriv=False)
    try:
        assert_array_almost_equal(expected_output, loss, decimal=3)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {loss}"


@pytest.mark.parametrize(
    "name, target, predicted, expected_output",
    [
        (
            "Case 1 : Binary Cross Entropy - Tensorflow example",
            np.array([0, 1, 0, 0]),
            np.array([0.6, 0.3, 0.2, 0.8]),
            np.array([2.5, -3.33333333, 1.25, 5]),
        )
    ],
)
def test_Loss_binary_cross_entropy_derivative(name, target, predicted, expected_output):
    loss = Loss.binary_cross_entropy(target, predicted, deriv=True)
    try:
        assert_allclose(expected_output, loss)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {loss}"


@pytest.mark.parametrize(
    "name, target, predicted, expected_output",
    [
        (
            "Case 1 : Cross Entropy (one hot encoded) 4 class",
            np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
            np.array([[0.25, 0.25, 0.25, 0.25], [0.01, 0.01, 0.01, 0.96]]),
            0.7135,
        ),
        (
            "Case 2 : Cross Entropy - Tensorflow example",
            np.array([[0, 1, 0], [0, 0, 1]]),
            np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]]),
            1.1769392,
        ),
    ],
)
def test_Loss_categorical_cross_entropy(name, target, predicted, expected_output):
    loss = Loss.categorical_cross_entropy(target, predicted, deriv=False)
    try:
        assert_array_almost_equal(expected_output, loss, decimal=3)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {loss}"


@pytest.mark.parametrize(
    "name, target, predicted, expected_output",
    [
        (
            "Case 1 : Cross Entropy (one hot encoded) 4 class",
            np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
            np.array([[0.25, 0.25, 0.25, 0.25], [0.01, 0.01, 0.01, 0.96]]),
            np.array(
                [
                    [1.33333333, 1.33333333, 1.33333333, -4],
                    [1.01010101, 1.01010101, 1.01010101, -1.04166667],
                ]
            ),
        )
    ],
)
def test_Loss_categorical_cross_entropy_derivative(
    name, target, predicted, expected_output
):
    loss = Loss.categorical_cross_entropy(target, predicted, deriv=True)
    try:
        assert_allclose(expected_output, loss)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {loss}"
