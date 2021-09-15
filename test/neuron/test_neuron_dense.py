import pytest
from src.neuron import NeuronDense


@pytest.mark.parametrize(
    "name, input_shape",
    [("Test 1 - Input shape 1 x 1", (1, 1)), ("Test 2 - Input shape 3 x 1", (3, 1))],
)
def test_dense_neuron(name, input_shape):
    neuron = NeuronDense(input_shape)
    assert neuron._weight_shape == input_shape, f"Wrong weight shape"
    assert neuron._weight_bias.shape == (1,), f"Wrong weight bias shape"
    assert neuron.weight.shape == (input_shape[0],), f"Wrong shape of weight matrix"
