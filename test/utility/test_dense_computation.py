import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal_nulp

from src.utility import dense_computation


def data1():
    return np.array([1, 0.05, 0.1, 0.25])


def data2():
    return np.array([1, 0.05, 0.1])


def weight1():
    return np.array(
        [
            np.array([0.35]),
            np.array([0.15]),
            np.array([0.2]),
            np.array([0.3]),
        ]
    )


def weight2():
    return np.array(
        [
            np.array([0.35, 0.35]),
            np.array([0.15, 0.25]),
            np.array([0.2, 0.3]),
        ]
    )


@pytest.mark.parametrize(
    "name, params, expected_output",
    [
        ("Test 1 - 1 neuron", (data1(), weight1()), np.array([0.4525])),
        ("Test 2 - 2 neuron", (data2(), weight2()), np.array([0.3775, 0.3925])),
    ],
)
def test_dense_computation(name, params, expected_output):
    out = dense_computation(*params)
    try:
        assert_array_almost_equal_nulp(out, expected_output)
    except:
        assert False, f"{expected_output} expected but got {out}"
