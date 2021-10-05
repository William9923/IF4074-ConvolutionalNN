import pytest
import numpy as np
from src.layer import MaxPooling2D
from src.optimizer import SGD

matrix = np.array(
    [
        np.array([1, 2, 3, 2, 3]),
        np.array([8, 3, 7, 4, 9]),
        np.array([9, 1, 4, 7, 2]),
        np.array([2, 6, 5, 3, 9]),
    ]
)

matrix2 = np.array(
    [
        np.array([1, 2, 3, 2, 3]),
        np.array([8, 3, 7, 4, 9]),
        np.array([9, 1, 4, 7, 2]),
    ]
)

matrix3 = np.array(
    [
        np.array([1, 2, 3, 2, 3, 4]),
        np.array([8, 3, 7, 4, 9, 4]),
        np.array([9, 1, 4, 7, 2, 10]),
        np.array([2, 6, 5, 3, 9, 1]),
        np.array([8, 3, 7, 4, 9, 3]),
        np.array([9, 1, 4, 7, 2, 12]),
        np.array([2, 6, 5, 3, 9, 10]),
    ]
)

matrix4 = np.array(
    [
        np.array([1, 2, 3, 2, 3, 4, 2, 3, 4, 1]),
        np.array([8, 3, 7, 4, 9, 4, 2, 3, 4, 1]),
        np.array([9, 1, 4, 7, 2, 10, 3, 10, 30, 3]),
        np.array([2, 6, 5, 3, 9, 1, 4, 6, 1, 3]),
        np.array([8, 3, 7, 4, 9, 3, 23, 11, 23, 33]),
        np.array([9, 1, 4, 7, 2, 12, 2, 3, 4, 5]),
        np.array([2, 6, 5, 3, 9, 10, 1, 1, 1, 1]),
    ]
)


def createBatch(data):
    batch = np.array(
        [
            data.copy(),
            data.copy(),
            data.copy(),
            data.copy(),
        ]
    )

    return batch


def stackData(matrix):
    data = np.stack(
        [
            matrix.copy(),
            matrix.copy(),
            matrix.copy(),
        ],
        axis=-1,
    )

    return data


def data(matrix):
    data = stackData(matrix)
    batch = createBatch(data)
    return batch


@pytest.mark.parametrize(
    "name, batch, params, expected_output",
    [
        (
            "Test 1 - Pool data 1",
            data(matrix),
            ((3, 3), (1, 1)),
            np.array(
                [
                    np.stack(
                        [
                            np.array(
                                [
                                    np.array([9.0, 7.0, 9.0]),
                                    np.array([9.0, 7.0, 9.0]),
                                ],
                            )
                            for _ in range(3)
                        ],
                        axis=-1,
                    )
                    for _ in range(4)
                ]
            ),
        ),
        (
            "Test 2 - Pool data 2",
            data(matrix2),
            ((2, 2), (1, 1)),
            np.array(
                [
                    np.stack(
                        [
                            np.array(
                                [
                                    np.array([8.0, 7.0, 7.0, 9.0]),
                                    np.array([9.0, 7.0, 7.0, 9.0]),
                                ],
                            )
                            for _ in range(3)
                        ],
                        axis=-1,
                    )
                    for _ in range(4)
                ]
            ),
        ),
        (
            "Test 3 - Pool data 3",
            data(matrix3),
            ((3, 3), (2, 2)),
            np.array(
                [
                    np.stack(
                        [
                            np.array(
                                [
                                    np.array([9.0, 9.0]),
                                    np.array([9.0, 9.0]),
                                    np.array([9.0, 9.0]),
                                ],
                            )
                            for _ in range(3)
                        ],
                        axis=-1,
                    )
                    for _ in range(4)
                ]
            ),
        ),
        (
            "Test 4 - Pool data 4",
            data(matrix4),
            ((3, 3), (2, 2)),
            np.array(
                [
                    np.stack(
                        [
                            np.array(
                                [
                                    np.array([9.0, 9.0, 10.0, 30.0]),
                                    np.array([9.0, 9.0, 23.0, 30.0]),
                                    np.array([9.0, 9.0, 23.0, 23.0]),
                                ],
                            )
                            for _ in range(3)
                        ],
                        axis=-1,
                    )
                    for _ in range(4)
                ]
            ),
        ),
    ],
)
def test_max_pooling_propagate(name, batch, params, expected_output):
    layer = MaxPooling2D(*params)
    out = layer.forward_propagation(batch)
    np.testing.assert_equal(out, expected_output)


@pytest.mark.parametrize(
    "name, batch, params, errors, expected_propagate_error_shape",
    [
        (
            "Test 1 - Pool data 1",
            data(matrix),
            ((3, 3), (1, 1)),
            np.ones([4, 2, 3, 3]),
            np.array([4, 4, 5, 3]),
        ),
        (
            "Test 2 - Pool data 2",
            data(matrix2),
            ((2, 2), (1, 1)),
            np.ones([4, 2, 4, 3]),
            np.array([4, 3, 5, 3]),
        ),
        (
            "Test 3 - Pool data 3",
            data(matrix3),
            ((3, 3), (2, 2)),
            np.ones([4, 3, 2, 3]),
            np.array([4, 7, 6, 3]),
        ),
        (
            "Test 4 - Pool data 4",
            data(matrix4),
            ((3, 3), (2, 2)),
            np.ones([4, 3, 4, 3]),
            np.array(data(matrix4).shape),
        ),
    ],
)
def test_max_pooling_backprop_shape(
    name, batch, params, errors, expected_propagate_error_shape
):
    layer = MaxPooling2D(*params)
    layer.build(batch.shape[1:])
    layer.forward_propagation(batch)
    propagate_error = layer.backward_propagation(SGD(), errors)
    np.testing.assert_array_almost_equal(
        expected_propagate_error_shape,
        propagate_error.shape,
        decimal=5,
        err_msg=f"{name} | Output shape -> Expected: {expected_propagate_error_shape}, Got: {propagate_error.shape}",
    )
