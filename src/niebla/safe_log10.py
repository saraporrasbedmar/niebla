import numpy as np


def log10_safe(array_input):
    array_copy = array_input.copy()
    array_copy[array_copy < 1e-43] = 1e-43
    array_copy = np.log10(array_copy)
    array_copy[np.isnan(array_copy)] = -43.
    array_copy[np.invert(np.isfinite(array_copy))] = -43.
    return array_copy
