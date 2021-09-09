import pytest
import numpy as np

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
            ((2, 2), (1, 1), 3),
            np.array(
                [
                    np.stack(
                        [
                            np.array(
                                [
                                    np.array([4.0, 4.0, 5.0, 2.0]),
                                    np.array([16.0, 20.0, 12.0, 29.0]),
                                    np.array([6.0, 8.0, 22.0, 4.0]),
                                ]
                            )
                            for _ in range(3)
                        ],
                        axis=-1,
                    )
                    for _ in range(4)
                ]
            ),
        )
    ],
)
def test_neuron_conv2d(name, batch, params, expected_output):
    neuron = NeuronConv2D(*params)
    neuron._kernels = np.array([kernel() for _ in range(neuron._input_shape)])
    out = neuron.compute(batch)
    assert (out == expected_output).all(), f'Wrong Output'
