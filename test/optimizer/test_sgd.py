import pytest
import numpy as np

from src.optimizer import SGD


@pytest.mark.parametrize(
    "name, opt, params, expected_output",
    [
        ("Test 1 - SGD Momentum 1", SGD(learning_rate=1, momentum=1), (1, 0.2), 0.8),
        ("Test 2 - SGD Momentum 0", SGD(learning_rate=1, momentum=0), (1, 0.2), 0.8),
    ],
)
def test_sgd_single(name, opt, params, expected_output):
    out = opt.update(*params)
    assert out == expected_output, "False Output"
