import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from src.layer.detector import ReLU


@pytest.mark.parametrize(
    "name, input_shape, expected_input, expected_output",
    [
        (
            "Test Build (Input Layer) - (None, 32, 32, 3)",
            (32, 32, 3),
            (32, 32, 3),
            (32, 32, 3),
        ),
        (
            "Test Build (Near Output Layer) - (None, 3, 3, 64)",
            (3, 3, 64),
            (3, 3, 64),
            (3, 3, 64),
        ),
    ],
)
def test_ReLU_build(name, input_shape, expected_input, expected_output):
    layer = ReLU()
    layer.build(input_shape)

    assert (
        layer.input_shape == expected_input
    ), f"{name} | Input shape -> Expected: {expected_input}, Got: {layer.input_shape}"
    assert (
        layer.output_shape == expected_output
    ), f"{name} | Output shape -> Expected: {expected_output}, Got: {layer.output_shape}"


@pytest.mark.parametrize(
    "name, input_shape, batch, expected_output",
    [
        (
            "Test - (None, 2, 2, 2)",
            (2, 2, 2),
            np.array(
                [
                    [
                        [2171, 2170],
                        [5954, 2064],
                    ],
                    [
                        [13042, 13575],
                        [11023, 6425],
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [2171, 2170],
                        [5954, 2064],
                    ],
                    [
                        [13042, 13575],
                        [11023, 6425],
                    ],
                ]
            ),
        ),
        (
            "Test - (None, 2, 2, 3)",
            (2, 2, 3),
            np.array(
                [
                    [
                        [2171, 2170],
                        [5954, 2064],
                    ],
                    [
                        [13042, 13575],
                        [11023, 6425],
                    ],
                    [
                        [-13042, -13575],
                        [-11023, -6425],
                    ]
                ]
            ),
            np.array(
                [
                    [
                        [2171, 2170],
                        [5954, 2064],
                    ],
                    [
                        [13042, 13575],
                        [11023, 6425],
                    ],
                    [
                        [0, 0],
                        [0, 0],
                    ],
                ]
            ),
        ),
    ],
)
def test_ReLU_forward_propagation(name, input_shape, batch, expected_output):
    layer = ReLU()
    layer.build(input_shape)

    output = layer.forward_propagation(batch)
    try:
        assert_array_almost_equal(expected_output, output, decimal=5)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {output}"
