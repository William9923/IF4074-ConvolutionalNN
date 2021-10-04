import pytest
import numpy as np

from src.optimizer import SGD


@pytest.mark.parametrize(
    "name, opt, params, expected_output",
    [
        ("Test 1 - SGD Momentum 1", SGD(learning_rate=1, momentum=1), (1, 0.2), 0.8),
        ("Test 2 - SGD Momentum 0", SGD(learning_rate=1, momentum=0), (1, 0.2), 0.8),
    ],
)
def test_sgd_single(name, opt, params, expected_output):
    out = opt.update(*params)
    assert out == expected_output, "False Output"


@pytest.mark.parametrize(
    "name, opt, params, expected_output",
    [
        (
            "Test 1 - SGD Momentum 1",
            SGD(learning_rate=1, momentum=1),
            (
                np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
                np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            ),
            np.array([[-3.5, -3.5, -3.5], [-2.5, -2.5, -2.5], [-1.5, -1.5, -1.5]]),
        ),
    ],
)
def test_sgd_matrix(name, opt, params, expected_output):
    out = opt.update_matrix(*params)
    assert (out == expected_output).all(), "False Output"
