import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from src.neuron import NeuronConv2D


def data():
    matrix = np.array(
        [
            np.array([1, 2, 3, 2, 3]),
            np.array([8, 3, 7, 4, 9]),
            np.array([9, 1, 4, 7, 2]),
            np.array([2, 6, 5, 3, 9]),
        ]
    )

    data = np.stack(
        [
            matrix.copy(),
            matrix.copy(),
            matrix.copy(),
        ],
        axis=-1,
    )

    batch = np.array(
        [
            data.copy(),
            data.copy(),
            data.copy(),
            data.copy(),
        ]
    )

    return batch


def kernel():
    return np.array(
        [
            np.array([1, 3]),
            np.array([0, -1]),
        ]
    )


@pytest.mark.parametrize(
    "name, batch, params, expected_output",
    [
        (
            "Test 1 - Compute 3 channels",
            data(),
            ((2, 2), (1, 1), (4, 5, 3)),
            np.array(
                [
                    np.array(
                        [
                            [13.0, 13.0, 16.0, 7.0], 
                            [49.0, 61.0, 37.0, 88.0], 
                            [19.0, 25.0, 67.0, 13.0]
                        ]
                    )
                    for _ in range(4)
                ]
            ),
        )
    ],
)
def test_neuron_conv2d_compute(name, batch, params, expected_output):
    neuron = NeuronConv2D(*params)
    channels = neuron._input_shape[2]
    neuron._kernels = np.stack([kernel() for _ in range(channels)], axis=-1)
    neuron._bias = 1
    out = neuron.compute(batch)
    assert_array_almost_equal(out, expected_output)
