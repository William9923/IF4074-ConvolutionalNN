import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from src.layer.detector import Softmax
from src.optimizer import SGD


@pytest.mark.parametrize(
    "name, input_shape, expected_input, expected_output",
    [
        (
            "Test Build (Input Layer) - (None, 10) -> After Dense",
            (10),
            (10),
            (10),
        ),
    ],
)
def test_Softmax_build(name, input_shape, expected_input, expected_output):
    layer = Softmax()
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
            "Test - (None, 5) -> Dense",
            (5),
            np.array([[-20, -1.0, 0.0, 1.0, 20]]),
            np.array(
                [
                    [
                        4.2483541e-18,
                        7.5825607e-10,
                        2.0611537e-09,
                        5.6027964e-09,
                        1.0000000e00,
                    ]
                ]
            ),
        ),
    ],
)
def test_Softmax_forward_propagation(name, input_shape, batch, expected_output):
    layer = Softmax()
    layer.build(input_shape)

    output = layer.forward_propagation(batch)
    try:
        assert_array_almost_equal(expected_output, output, decimal=5)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {output}"

    sums = np.sum(output, axis=1)
    expects = np.ones(output.shape[0])
    try:
        assert_array_almost_equal(expects, sums, decimal=5)
    except AssertionError as err:
        assert False, f"{name} | {expects} expected, but get {sums}"


@pytest.mark.parametrize(
    "name, input_shape, batch, errors, expected_propagate_error_shape",
    [
        (
            "Test - (None, 5) -> Dense",
            (5),
            np.array([[-20, -1.0, 0.0, 1.0, 20]]),
            np.array(
                [
                    [
                        4.2483541e-18,
                        7.5825607e-10,
                        2.0611537e-09,
                        5.6027964e-09,
                        1.0000000e00,
                    ]
                ]
            ),
            np.array([1, 5]),
        ),
    ],
)
def test_Softmax_backward_propagation_shape(
    name, input_shape, batch, errors, expected_propagate_error_shape
):
    layer = Softmax()
    layer.build(input_shape)

    layer.forward_propagation(batch)
    propagate_error = layer.backward_propagation(SGD(), errors)

    assert_array_almost_equal(
        expected_propagate_error_shape,
        propagate_error.shape,
        decimal=5,
        err_msg=f"{name} | Output shape -> Expected: {expected_propagate_error_shape}, Got: {propagate_error.shape}",
    )
