import pytest
import numpy as np
from src.layer import Dense


def data1():
    return np.array([np.array([1, 0.05, 0.1, 0.25])])


def data2():
    return np.array(
        [
            np.array([1, 0.05, 0.1, 0.25]),
            np.array([1, 0.05, 0.1, 0.25]),
            np.array([1, 0.05, 0.1, 0.25]),
        ]
    )


@pytest.mark.parametrize(
    "name, params, batch, expected_output_shape",
    [
        ("Test 1 - 1 batch 1 unit", (1, 3), data1(), (1, 1)),
        ("Test 2 - 1 batch 2 unit", (2, 3), data1(), (1, 2)),
        ("Test 3 - 3 batch 1 unit", (1, 3), data2(), (3, 1)),
        ("Test 4 - 3 batch 3 unit", (3, 3), data2(), (3, 3)),
    ],
)
def test_dense_layer(name, params, batch, expected_output_shape):
    dense_layer = Dense(*params)
    out = dense_layer.forward_propagation(batch)
    print(out)
    assert out.shape == expected_output_shape, f"Wrong output shape"
    assert dense_layer.output_shape == (params[0], 1)
