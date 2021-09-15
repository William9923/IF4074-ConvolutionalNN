import pytest

from src.utility import calc_params_conv, calc_params_dense


@pytest.mark.parametrize(
    "name, params, expected_output",
    [("Test 1 - Test calculation dense", (10, 15), 165)],
)
def test_calc_dense(name, params, expected_output):
    out = calc_params_dense(*params)
    assert out == expected_output, "Wrong output"


@pytest.mark.parametrize(
    "name, params, expected_output",
    [("Test 1 - Test calculation conv", (36, 36, 3, 10), 38890)],
)
def test_calc_conv(name, params, expected_output):
    out = calc_params_conv(*params)
    assert out == expected_output, "Wrong output"
