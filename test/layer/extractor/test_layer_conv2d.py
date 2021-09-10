import pytest
import numpy as np

from src.layer import Conv2D


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
            "Test 1 - Forward 10 neurons",
            data(),
            (10, (2, 2), (1, 1), (0,0,0,0), (4, 5, 3)),
            np.array(
                [
                    np.stack(
                        [
                            np.array(
                                [
                                    np.array([5.0, 5.0, 6.0, 3.0]) * 3,
                                    np.array([17.0, 21.0, 13.0, 30.0]) * 3,
                                    np.array([7.0, 9.0, 23.0, 5.0]) * 3,
                                ]
                            )
                            for _ in range(10)
                        ],
                        axis=-1,
                    )
                    for _ in range(4)
                ]
            ),
        )
    ],
)
def test_forward_layer_conv2d(name, batch, params, expected_output):
    layer = Conv2D(*params)
    channels = layer.input_shape[2]
    for neuron in layer._neurons:
        neuron._kernels = np.array([kernel() for _ in range(channels)])
        neuron._bias = np.ones(neuron._output_shape)
    out = layer.forward_propagation(batch)

    assert (out == expected_output).all(), f"Wrong {out.shape} {expected_output.shape}"
