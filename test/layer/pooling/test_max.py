import pytest
import numpy as np
from src.layer import MaxPooling2D

matrix = np.array(
        [
            np.array([1, 2, 3, 2, 3]),
            np.array([8, 3, 7, 4, 9]),
            np.array([9, 1, 4, 7, 2]),
            np.array([2, 6, 5, 3, 9]),
        ])
    
matrix2 = np.array(
        [
            np.array([1, 2, 3, 2, 3]),
            np.array([8, 3, 7, 4, 9]),
            np.array([9, 1, 4, 7, 2]),
        ])

matrix3 = np.array(   
        [
            np.array([1, 2, 3, 2, 3, 4]),
            np.array([8, 3, 7, 4, 9, 4]),
            np.array([9, 1, 4, 7, 2, 10]),
            np.array([2, 6, 5, 3, 9, 1]),
            np.array([8, 3, 7, 4, 9, 3]),
            np.array([9, 1, 4, 7, 2, 12]),
            np.array([2, 6, 5, 3, 9, 10]),
        ])
    
matrix4 = np.array(
        [
            np.array([1, 2, 3, 2, 3, 4, 2, 3, 4, 1]),
            np.array([8, 3, 7, 4, 9, 4, 2, 3, 4,1 ]),
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
            ((3, 3),(1, 1)),
            np.array(
                [
                    np.array(
                        [
                            np.array(
                                [
                                    np.array([9.0, 9.0, 9.0]),
                                    np.array([7.0, 7.0, 7.0]),
                                    np.array([9.0, 9.0, 9.0]),
                                ],
                            )
                            for _ in range(2)
                        ],
                    )
                    for _ in range(4)
                ]
            ),
        )
    ],
)
def test_max_pooling_data1(name, batch, params, expected_output):
    layer = MaxPooling2D(*params)
    out = layer.propagate(batch)
    np.testing.assert_equal(out, expected_output)


@pytest.mark.parametrize(
    "name, batch, params, expected_output",
    [
        (
            "Test 2 - Pool data 2",
            data(matrix2),
            ((2, 2),(1, 1)),
            np.array(
                [
                    np.array(
                        [
                            np.array(
                                [
                                    np.array([8.0, 8.0, 8.0]),
                                    np.array([7.0, 7.0, 7.0]),
                                    np.array([7.0, 7.0, 7.0]),
                                    np.array([9.0, 9.0, 9.0]),
                                ],
                            ),
                            np.array(
                                [
                                    np.array([9.0, 9.0, 9.0]),
                                    np.array([7.0, 7.0, 7.0]),
                                    np.array([7.0, 7.0, 7.0]),
                                    np.array([9.0, 9.0, 9.0]),
                                ]
                            )
                        ],
                    )
                    for _ in range(4)
                ]
            ),
        )
    ],
)
def test_max_pooling_data2(name, batch, params, expected_output):
    layer = MaxPooling2D(*params)
    out = layer.propagate(batch)
    np.testing.assert_equal(out, expected_output)


@pytest.mark.parametrize(
    "name, batch, params, expected_output",
    [
        (
            "Test 3 - Pool data 3",
            data(matrix3),
            ((3, 3),(2, 2)),
            np.array(
                [
                    np.array(
                        [
                            np.array(
                                [
                                    np.array([9.0, 9.0, 9.0]),
                                    np.array([9.0, 9.0, 9.0]),
                                ],
                            )
                            for _ in range(3)
                        ],
                    )
                    for _ in range(4)
                ]
            ),
        )
    ],
)
def test_max_pooling_data3(name, batch, params, expected_output):
    layer = MaxPooling2D(*params)
    out = layer.propagate(batch)
    np.testing.assert_equal(out, expected_output)

@pytest.mark.parametrize(
    "name, batch, params, expected_output",
    [
        (
            "Test 4 - Pool data 4",
            data(matrix4),
            ((3, 3),(2, 2)),
            np.array(
                [
                    np.array(
                        [
                            np.array(
                                [
                                    np.array([9.0, 9.0, 9.0]),
                                    np.array([9.0, 9.0, 9.0]),
                                    np.array([10.0, 10.0, 10.0]),
                                    np.array([30.0, 30.0, 30.0]),
                                ],
                            ),
                            np.array(
                                [
                                    np.array([9.0, 9.0, 9.0]),
                                    np.array([9.0, 9.0, 9.0]),
                                    np.array([23.0, 23.0, 23.0]),
                                    np.array([30.0, 30.0, 30.0]),
                                ],
                            ),
                            np.array(
                                [
                                    np.array([9.0, 9.0, 9.0]),
                                    np.array([9.0, 9.0, 9.0]),
                                    np.array([23.0, 23.0, 23.0]),
                                    np.array([23.0, 23.0, 23.0]),
                                ],
                            ),
                            
                        ],
                    )
                    for _ in range(4)
                ]
            ),
        )
    ],
)
def test_max_pooling_data4(name, batch, params, expected_output):
    layer = MaxPooling2D(*params)
    out = layer.propagate(batch)
    np.testing.assert_equal(out, expected_output)
