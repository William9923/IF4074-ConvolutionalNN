from numpy.testing._private.utils import assert_array_almost_equal
import pytest
import numpy as np

from src.layer import Conv2D
from src.optimizer import SGD


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


def college_data():
    c1 = np.array([[16, 24, 32], [47, 18, 26], [68, 12, 9]])
    c2 = np.array([[26, 57, 43], [24, 21, 12], [2, 11, 19]])
    c3 = np.array([[18, 47, 21], [4, 6, 12], [81, 22, 13]])
    data = np.stack([c1, c2, c3], axis=-1)
    return np.array([data])


def kernel():
    return np.array(
        [
            np.array([1, 3]),
            np.array([0, -1]),
        ]
    )


def kernel_college():
    return np.array(
        [
            [[0, -1], [1, 0]],
            [[5, 4], [3, 2]],
            [[16, 24], [68, -2]],
            [[60, 22], [32, 18]],
            [[35, 46], [7, 23]],
            [[78, 81], [20, 42]],
        ]
    )


@pytest.mark.parametrize(
    "name, batch, params, expected_output",
    [
        (
            "Test 1 - Forward 10 neurons",
            data(),
            (10, (2, 2), (1, 1), (0, 0, 0, 0), (4, 5, 3)),
            np.array(
                [
                    np.stack(
                        [
                            np.array(
                                [
                                    np.array([13.0, 13.0, 16.0, 7.0]),
                                    np.array([49.0, 61.0, 37.0, 88.0]),
                                    np.array([19.0, 25.0, 67.0, 13.0]),
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
        neuron._kernels = np.stack([kernel() for _ in range(channels)], axis=-1)
        neuron._bias = 1
    out = layer.forward_propagation(batch)

    assert_array_almost_equal(out, expected_output)


def test_forward_layer_with_college_data():
    layer = Conv2D(2, (2, 2), (1, 1), input_shape=(3, 3, 3))
    kernel = kernel_college()
    layer._neurons[0]._kernels[:, :, 0] = kernel[0]
    layer._neurons[0]._kernels[:, :, 1] = kernel[1]
    layer._neurons[0]._kernels[:, :, 2] = kernel[2]

    layer._neurons[1]._kernels[:, :, 0] = kernel[3]
    layer._neurons[1]._kernels[:, :, 1] = kernel[4]
    layer._neurons[1]._kernels[:, :, 2] = kernel[5]

    layer._neurons[0]._bias = 0
    layer._neurons[1]._bias = 0

    out = layer.forward_propagation(college_data())
    expected_output = np.array(
        [
            np.stack(
                [
                    [
                        [2171.0, 2170.0],
                        [5954.0, 2064.0],
                    ],
                    [[13042.0, 13575.0], [11023.0, 6425.0]],
                ],
                axis=-1,
            )
        ]
    )
    assert_array_almost_equal(out, expected_output)


@pytest.mark.parametrize(
    "name, batch, errors, layer_params, expected_shape",
    [
        (
            "Test 1 - Test Shape",
            np.random.rand(2, 7, 7, 5),
            np.random.rand(2, 5, 5, 3),
            (3, (3, 3), (1, 1), (0, 0, 0, 0), (7, 7, 5)),
            (2, 7, 7, 5),
        )
    ],
)
def test_backward(name, batch, errors, layer_params, expected_shape):
    layer = Conv2D(*layer_params)
    layer.forward_propagation(batch)
    new_err = layer.backward_propagation(SGD(), errors)
    assert new_err.shape == expected_shape, "False Shape Output"
