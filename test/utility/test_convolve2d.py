import pytest
import numpy as np

from src.utility import convolve2D

def data():
    return np.array(
        [
            np.array([1, 2, 3, 2, 3]),
            np.array([8, 3, 7, 4, 9]),
            np.array([9, 1, 4, 7, 2]),
            np.array([2, 6, 5, 3, 9]),
        ]
    )

def kernel():
    return np.array(
        [
            np.array([1, 3]),
            np.array([0, -1]),
        ]
    )

def test_stride_1_1():
    stride = (1, 1)
    params = (data(), kernel(), stride)
    expected = np.array(
        [
            np.array([4., 4., 5., 2.]),
            np.array([16., 20., 12., 29.]),
            np.array([6., 8., 22., 4.]),
        ]
    )
    out = convolve2D(*params)

    assert (expected == out).all(), 'Wrong output'

def test_stride_2_1():
    stride = (2, 1)
    params = (data(), kernel(), stride)
    expected = np.array(
        [
            np.array([4., 4., 5., 2.]),
            np.array([6., 8., 22., 4.])
        ]
    )
    out = convolve2D(*params)

    assert (expected == out).all(), 'Wrong output'

def test_stride_3_1():
    stride = (3, 1)
    params = (data(), kernel(), stride)
    expected = np.array(
        [
            np.array([4., 4., 5., 2.]),
        ]
    )
    out = convolve2D(*params)

    assert (expected == out).all(), 'Wrong output'

def test_stride_1_2():
    stride = (1, 2)
    params = (data(), kernel(), stride)
    expected = np.array(
        [
            np.array([4., 5.]),
            np.array([16., 12.]),
            np.array([6., 22.]),
        ]
    )
    out = convolve2D(*params)

    assert (expected == out).all(), 'Wrong output'

def test_stride_1_4():
    stride = (1, 4)
    params = (data(), kernel(), stride)
    expected = np.array(
        [
            np.array([4.]),
            np.array([16.]),
            np.array([6.]),
        ]
    )
    out = convolve2D(*params)

    assert (expected == out).all(), 'Wrong output'