import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from src.utility import (
    normalize_result,
    calc_convoluted_shape,
    calc_input_shape_with_padding,
)


@pytest.mark.parametrize(
    "name, input, expected_output",
    [
        (
            "Test Accuracy - 100%",
            np.array(
                [
                    [0.2, 0.3, 0.5],
                    [0.1, 0.1, 0.8],
                    [0.2, 0.1, 0.7],
                    [0.5, 0.1, 0.4],
                    [0.4, 0.4, 0.2],
                ]
            ),
            np.array(
                [
                    2,
                    2,
                    2,
                    0,
                    0,
                ]
            ),
        ),
    ],
)
def test_Utility_normalize_result(name, input, expected_output):
    res = normalize_result(input)
    try:
        assert_array_equal(expected_output, res)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {res}"


@pytest.mark.parametrize(
    "name, params, expected_output",
    [("Test Padding Input Shape", ((10, 10, 10), ((0, 1, 2, 3))), ((11, 15, 10)))],
)
def test_Utility_calc_input_shape_with_padding(name, params, expected_output):
    out = calc_input_shape_with_padding(*params)
    assert out == expected_output, "Wrong output"


@pytest.mark.parametrize(
    "name, params, expected_output",
    [
        ("Test with Stride 2 2", ((10, 10, 10), (3, 3), (2, 2)), ((4, 4, 10))),
        ("Test with Stride 1 1", ((10, 10, 10), (3, 3), (1, 1)), (8, 8, 10)),
    ],
)
def test_Utility_calc_convoluted_shape(name, params, expected_output):
    out = calc_convoluted_shape(*params)
    assert out == expected_output, "Wrong output"
