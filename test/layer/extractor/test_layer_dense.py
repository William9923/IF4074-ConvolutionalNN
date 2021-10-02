import pytest
import numpy as np

from src.layer import Dense


def data():
    return np.array([[1, 0.95, 0.1, 0.25], [0.75, 0.4, 0.3, 0.2]])


@pytest.mark.parametrize(
    "name, params, batch, expected_output_shape",
    [
        ("Test 1 - 2 batch 2 unit", (2, 4), data(), (2, 2)),
    ],
)
def test_dense_layer(name, params, batch, expected_output_shape):
    dense_layer = Dense(*params)
    out = dense_layer.forward_propagation(batch)
    assert out.shape == expected_output_shape, f"Wrong output shape"
