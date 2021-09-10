import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from src.utility import normalize_result

@pytest.mark.parametrize("name,  input, expected_output", [
    (
        "Test Normalize Result",
        np.array([
            [0.2,0.3,0.5],
            [0.1,0.1,0.8],
            [0.2,0.1,0.7],
            [0.5,0.1,0.4],
            [0.4,0.4,0.2],
        ]),
        np.array([
            2,
            2,
            2,
            0,
            0,
        ]),
    ),
])
def test_Utility_normalize_result(name, input, expected_output):
    res = normalize_result(input)
    print(res)
    try :
        assert_array_equal(expected_output, res)
    except AssertionError as err:
        assert False, f"{name} | {expected_output} expected, but get {res}"


