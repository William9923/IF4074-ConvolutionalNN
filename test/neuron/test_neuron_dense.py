import pytest
import numpy as np

from src.neuron import NeuronDense
from src.optimizer import SGD


def data():
    return np.array([[1, 0.95, 0.1, 0.25], [0.75, 0.4, 0.3, 0.2]])


def weights():
    return np.array([1, 2, 3, 4])


@pytest.mark.parametrize(
    "name, input_shape",
    [("Test 1 - Input shape 1", 1), ("Test 2 - Input shape 3", 3)],
)
def test_build_dense_neuron(name, input_shape):
    neuron = NeuronDense(input_shape)
    assert neuron._weights.shape == (input_shape,), f"Wrong weight shape"


@pytest.mark.parametrize(
    "name, input_shape, batch, expected_output",
    [("Test 1", 4, data(), np.array([5.2, 4.25]))],
)
def test_compute_dense_neuron(name, input_shape, batch, expected_output):
    neuron = NeuronDense(input_shape)
    neuron._weights = weights()
    neuron._bias = 1
    out = neuron.compute(batch)

    assert (out == expected_output).all(), f"Wrong Output"


@pytest.mark.parametrize(
    "name, batch, errors, neuron_params, expected_shape",
    [
        (
            "Test 1 - Test Shape",
            np.random.rand(2, 4),
            np.random.rand(2),
            4,
            (2, 4),
        )
    ],
)
def test_neuron_dense_backprop(name, batch, errors, neuron_params, expected_shape):
    neuron = NeuronDense(neuron_params)
    neuron.compute(batch)
    new_err = neuron.update_weights(SGD(), errors)
    assert (
        new_err.shape == expected_shape
    ), f"Expected {expected_shape} got {new_err.shape}"
