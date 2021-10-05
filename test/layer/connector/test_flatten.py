import pytest
import numpy as np

from src.layer.connector import Flatten
from src.optimizer import SGD


@pytest.mark.parametrize(
    "name, input_shape, expected_input, expected_output",
    [
        (
            "Test Build (Input Layer) - (None, 32, 32, 3)",
            (32, 32, 3),
            (32, 32, 3),
            3072,
        ),
        (
            "Test Build (Near Output Layer) - (None, 3, 3, 64)",
            (3, 3, 64),
            (3, 3, 64),
            576,
        ),
    ],
)
def test_Flatten_build(name, input_shape, expected_input, expected_output):
    layer = Flatten()
    layer.build(input_shape)

    assert (
        layer.input_shape == expected_input
    ), f"{name} | Input shape -> Expected: {expected_input}, Got: {layer.input_shape}"
    assert (
        layer.output_shape == expected_output
    ), f"{name} | Output shape -> Expected: {expected_output}, Got: {layer.output_shape}"


@pytest.mark.parametrize(
    "name, input_shape, batch, expected_output_shape",
    [
        (
            "Test Forward Prop (Input Layer) - (None, 32, 32, 3)",
            (
                32,
                32,
                3,
            ),
            np.random.random([100, 32, 32, 3]),
            (100, 3072),
        ),
        (
            "Test Forward Prop (Near Output Layer) - (None, 3, 3, 64)",
            (3, 3, 64),
            np.random.random([100, 3, 3, 64]),
            (100, 576),
        ),
    ],
)
def test_Flatten_forward_propagation(name, input_shape, batch, expected_output_shape):
    layer = Flatten()
    layer.build(input_shape)

    output = layer.forward_propagation(batch)
    assert (
        output.shape == expected_output_shape
    ), f"{name} Expected: {expected_output_shape}, Got: {output.shape}"


@pytest.mark.parametrize(
    "name, input_shape, error_batch, expected_reverse_shape",
    [
        (
            "Test Forward Prop (Input Layer) - (None, 32, 32, 3)",
            (
                32,
                32,
                3,
            ),
            np.random.random([100, 3072]),
            (100, 32, 32, 3),
        ),
        (
            "Test Forward Prop (Near Output Layer) - (None, 3, 3, 64)",
            (3, 3, 64),
            np.random.random([100, 576]),
            (100, 3, 3, 64),
        ),
    ],
)
def test_Flatten_backward_propagation(
    name, input_shape, error_batch, expected_reverse_shape
):
    layer = Flatten()
    layer.build(input_shape)

    reverse_output = layer.backward_propagation(SGD(), error_batch)
    assert (
        reverse_output.shape == expected_reverse_shape
    ), f"{name} Expected: {expected_reverse_shape}, Got: {reverse_output.shape}"
