import pytest
import numpy as np

from src.utility import pad2D

def data():
    return np.array(
        [
            np.array([1, 2, 3, 2, 3]),
            np.array([8, 3, 7, 4, 9]),
            np.array([9, 1, 4, 7, 2]),
            np.array([2, 6, 5, 3, 9]),
        ]
    )

def test_pad_top_1():
    pad = (1, 0, 0, 0)
    params = (data(), pad, 0)
    expected = np.array(
        [
            np.array([0, 0, 0, 0, 0]),
            np.array([1, 2, 3, 2, 3]),
            np.array([8, 3, 7, 4, 9]),
            np.array([9, 1, 4, 7, 2]),
            np.array([2, 6, 5, 3, 9]),
        ]
    )
    out = pad2D(*params)

    assert (expected == out).all(), 'Wrong output'

def test_pad_bot_1():
    pad = (0, 1, 0, 0)
    params = (data(), pad, 0)
    expected = np.array(
        [
            np.array([1, 2, 3, 2, 3]),
            np.array([8, 3, 7, 4, 9]),
            np.array([9, 1, 4, 7, 2]),
            np.array([2, 6, 5, 3, 9]),
            np.array([0, 0, 0, 0, 0]),
        ]
    )
    out = pad2D(*params)

    assert (expected == out).all(), 'Wrong output'

def test_pad_left_1():
    pad = (0, 0, 1, 0)
    params = (data(), pad, 0)
    expected = np.array(
        [
            np.array([0, 1, 2, 3, 2, 3]),
            np.array([0, 8, 3, 7, 4, 9]),
            np.array([0, 9, 1, 4, 7, 2]),
            np.array([0, 2, 6, 5, 3, 9]),
        ]
    )
    out = pad2D(*params)

    assert (expected == out).all(), 'Wrong output'

def test_pad_right_1():
    pad = (0, 0, 0, 1)
    params = (data(), pad, 0)
    expected = np.array(
        [
            np.array([1, 2, 3, 2, 3, 0]),
            np.array([8, 3, 7, 4, 9, 0]),
            np.array([9, 1, 4, 7, 2, 0]),
            np.array([2, 6, 5, 3, 9, 0]),
        ]
    )
    out = pad2D(*params)

    assert (expected == out).all(), 'Wrong output'

def test_pad_all_3():
    pad = (3, 3, 3, 3)
    params = (data(), pad, 0)
    expected = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 1, 2, 3, 2, 3, 0, 0, 0]),
            np.array([0, 0, 0, 8, 3, 7, 4, 9, 0, 0, 0]),
            np.array([0, 0, 0, 9, 1, 4, 7, 2, 0, 0, 0]),
            np.array([0, 0, 0, 2, 6, 5, 3, 9, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ]
    )
    out = pad2D(*params)

    assert (expected == out).all(), 'Wrong output'