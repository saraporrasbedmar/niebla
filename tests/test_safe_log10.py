import numpy as np
import pytest

from src.niebla.safe_log10 import log10_safe


def test_log10_safe_basic_values():
    arr = np.array([1.0, 10.0, 100.0])
    result = log10_safe(arr)

    expected = np.array([0.0, 1.0, 2.0])
    assert np.allclose(result, expected)

def test_log10_safe_clamps_small_values():
    arr = np.array([1e-50, 1e-43, 1e-42])
    result = log10_safe(arr)

    expected = np.array([-43.0, -43.0, np.log10(1e-42)])
    assert np.allclose(result, expected)

def test_log10_safe_handles_zero():
    arr = np.array([0.0])
    result = log10_safe(arr)

    assert result.shape == (1,)
    assert result[0] == -43.0

def test_log10_safe_handles_negative_values():
    arr = np.array([-1.0, -100.0])
    result = log10_safe(arr)

    assert np.all(result == -43.0)

def test_log10_safe_handles_nan():
    arr = np.array([np.nan, 1.0])
    result = log10_safe(arr)

    assert result[0] == -43.0
    assert result[1] == 0.0

def test_log10_safe_handles_inf_values():
    arr = np.array([np.inf, -np.inf, 10.0])
    result = log10_safe(arr)

    assert result[0] == -43.0
    assert result[1] == -43.0
    assert result[2] == 1.0

def test_log10_safe_does_not_modify_input():
    arr = np.array([0.0, 1.0, np.nan, -5.0])
    arr_copy = arr.copy()

    _ = log10_safe(arr)

    # Input must remain unchanged
    assert np.array_equal(arr, arr_copy, equal_nan=True)
