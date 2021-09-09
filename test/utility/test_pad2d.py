import pytest
import numpy as np

from src.utility import pad2D


def data():
    return np.array(
        [
            np.array([1, 2, 3, 2, 3]),
            np.array([8, 3, 7, 4, 9]),
            np.array([9, 1, 4, 7, 2]),
            np.array([2, 6, 5, 3, 9]),
        ]
    )


@pytest.mark.parametrize(
    "name, params, expected_output",
    [
        (
            "Test 1 - Pad Top 1",
            (data(), (1, 0, 0, 0), 0),
            np.array(
                [
                    np.array([0, 0, 0, 0, 0]),
                    np.array([1, 2, 3, 2, 3]),
                    np.array([8, 3, 7, 4, 9]),
                    np.array([9, 1, 4, 7, 2]),
                    np.array([2, 6, 5, 3, 9]),
                ]
            ),
        ),
        (
            "Test 2 - Pad Bot 1",
            (data(), (0, 1, 0, 0), 0),
            np.array(
                [
                    np.array([1, 2, 3, 2, 3]),
                    np.array([8, 3, 7, 4, 9]),
                    np.array([9, 1, 4, 7, 2]),
                    np.array([2, 6, 5, 3, 9]),
                    np.array([0, 0, 0, 0, 0]),
                ]
            ),
        ),
        (
            "Test 3 - Pad Left 1",
            (data(), (0, 0, 1, 0), 0),
            np.array(
                [
                    np.array([0, 1, 2, 3, 2, 3]),
                    np.array([0, 8, 3, 7, 4, 9]),
                    np.array([0, 9, 1, 4, 7, 2]),
                    np.array([0, 2, 6, 5, 3, 9]),
                ]
            ),
        ),
        (
            "Test 4 - Pad Right 1",
            (data(), (0, 0, 0, 1), 0),
            np.array(
                [
                    np.array([1, 2, 3, 2, 3, 0]),
                    np.array([8, 3, 7, 4, 9, 0]),
                    np.array([9, 1, 4, 7, 2, 0]),
                    np.array([2, 6, 5, 3, 9, 0]),
                ]
            ),
        ),
        (
            "Test 5 - Pad All 3",
            (data(), (3, 3, 3, 3), 0),
            np.array(
                [
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 1, 2, 3, 2, 3, 0, 0, 0]),
                    np.array([0, 0, 0, 8, 3, 7, 4, 9, 0, 0, 0]),
                    np.array([0, 0, 0, 9, 1, 4, 7, 2, 0, 0, 0]),
                    np.array([0, 0, 0, 2, 6, 5, 3, 9, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                ]
            ),
        ),
    ],
)
def test_pad2D(name, params, expected_output):
    out = pad2D(*params)
    assert (expected_output == out).all(), "Wrong output"
