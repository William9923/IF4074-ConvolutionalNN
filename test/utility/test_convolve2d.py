import pytest
import numpy as np

from src.utility import convolve2D


def data():
    return np.array(
        [
            np.array([1, 2, 3, 2, 3]),
            np.array([8, 3, 7, 4, 9]),
            np.array([9, 1, 4, 7, 2]),
            np.array([2, 6, 5, 3, 9]),
        ]
    )


def kernel():
    return np.array(
        [
            np.array([1, 3]),
            np.array([0, -1]),
        ]
    )


@pytest.mark.parametrize(
    "name, params, expected_output",
    [
        (
            "Test 1 - Stride 1-1",
            (data(), kernel(), (1, 1)),
            np.array(
                [
                    np.array([4.0, 4.0, 5.0, 2.0]),
                    np.array([16.0, 20.0, 12.0, 29.0]),
                    np.array([6.0, 8.0, 22.0, 4.0]),
                ]
            ),
        ),
        (
            "Test 2 - Stride 2-1",
            (data(), kernel(), (2, 1)),
            np.array([np.array([4.0, 4.0, 5.0, 2.0]), np.array([6.0, 8.0, 22.0, 4.0])]),
        ),
        (
            "Test 3 - Stride 3-1",
            (data(), kernel(), (3, 1)),
            np.array(
                [
                    np.array([4.0, 4.0, 5.0, 2.0]),
                ]
            ),
        ),
        (
            "Test 4 - Stride 1-2",
            (data(), kernel(), (1, 2)),
            np.array(
                [
                    np.array([4.0, 5.0]),
                    np.array([16.0, 12.0]),
                    np.array([6.0, 22.0]),
                ]
            ),
        ),
        (
            "Test 5 - Stride 1-4",
            (data(), kernel(), (1, 4)),
            np.array(
                [
                    np.array([4.0]),
                    np.array([16.0]),
                    np.array([6.0]),
                ]
            ),
        ),
    ],
)
def test_convolve2D(name, params, expected_output):
    out = convolve2D(*params)
    assert (expected_output == out).all(), "Wrong output"
