import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from src.activation import Activation

@pytest.mark.parametrize("name, input, expected_output", [
    (
        "Test 1 - Positive Number",
        1,
        1,
    ),
    (
        "Test 2 - Negative Number",
        -1,
        0,
    ),
    (
        "Test 3 - Sequence of positive & negative number",
        np.array([1.0,-1.0,2.0,-4.0]),
        np.array([1.0,0.0,2.0,0.0])
    ),
])
def test_Activation_ReLU(name, input, expected_output):
    result = Activation.relu(input)
    try :
        assert_array_almost_equal(expected_output, result, decimal=5)
    except AssertionError as err:
        assert False, f"{expected_output} expected, but get {result}"

@pytest.mark.parametrize("name, input, expected_output", [
    (
        "Test 1 - Positive Number",
        1,
        1,
    ),
    (
        "Test 2 - Negative Number",
        -1,
        0,
    ),
    (
        "Test 3 - Sequence of positive & negative number",
        np.array([1.0,-1.0,2.0,-4.0]),
        np.array([1.0,0.0,1.0,0.0])
    ),
])
def test_Activation_ReLU_derivative(name, input, expected_output):
    result = Activation.relu(input, deriv=True)
    try :
        assert_array_almost_equal(expected_output, result, decimal=5)
    except AssertionError as err:
        assert False, f"{expected_output} expected, but get {result}"


@pytest.mark.parametrize("name, input, expected_output", [
    (
        "Test 1 - Number",
        1,
        0.7310585786300049,
    ),
    (
        "Test 2 - Batch",
        np.array([1,2,3,-1,-2,4]),
        np.array([0.7310585786300049, 0.8807970779778823, 0.9525741268224334, 0.2689414213699951, 0.11920292202211755, 0.9820137900379085]),
    )
])
def test_Activation_Sigmoid(name, input, expected_output):
    result = Activation.sigmoid(input)
    try :
        assert_array_almost_equal(expected_output, result, decimal=5)
    except AssertionError as err:
        assert False, f"{expected_output} expected, but get {result}"  


@pytest.mark.parametrize("name, input, expected_output", [
    (
        "Test 1 - Number",
        1,
        0.19661193324148185,
    ),
    (
        "Test 2 - Batch",
        np.array([1,2,3,-1,-2,4]),
        np.array([0.19661193, 0.10499359, 0.04517666, 0.19661193, 0.10499359, 0.01766271]),
    )
])
def test_Activation_Sigmoid_derivative(name, input, expected_output):
    result = Activation.sigmoid(input, deriv=True)
    try :
        assert_array_almost_equal(expected_output, result, decimal=5)
    except AssertionError as err:
        assert False, f"{expected_output} expected, but get {result}"

@pytest.mark.parametrize("name, input, expected_output", [
    (
        "Test 1 - 1 Sequence Input",
        np.array([2,3,-1,0,1]),
        np.array([0.23412165725274, 0.63640864655883, 0.01165623095604, 0.031684920796124, 0.086128544436269])
    )
])
def test_Activation_Softmax(name, input, expected_output):
    result = Activation.softmax(input)
    try :
        assert_array_almost_equal(expected_output, result, decimal=5)
    except AssertionError as err:
        assert False, f"{expected_output} expected, but get {result}"

