import numpy as np
import pytest

from src.niebla.sfr_models import (
    sfr_model, _sfr_madau14, _sfr_cuba, _sfr_finke22a)



def test_dispatch_builtin_madau14():
    z = np.array([0.0, 1.0, 2.0])
    assert np.allclose(
        sfr_model(z, "sfr_madau14", verbose=False),
        _sfr_madau14(z, sfr_params=None, verbose=False)
    )

def test_constant_default_and_param():
    z_list = [0.0, 1.0, 2.0]
    out0 = sfr_model(z_list, "sfr_constant", verbose=False)
    assert np.allclose(out0, np.ones_like(np.array(z_list)) * 1.0)

    z = np.array([0.0, 1.0])
    out = sfr_model(z, "sfr_constant", sfr_params=3.5,
                    verbose=False)
    assert np.allclose(out, np.ones_like(z) * 3.5)

def test_madau14_scalar_matches_formula():
    z = 0.3
    out = _sfr_madau14(z, sfr_params=None, verbose=False)
    p0, p1, p2, p3 = 0.015, 2.7, 2.9, 5.6
    expected = p0 * (1+z)**p1 / (1 + ((1+z)/p2)**p3)
    assert np.isclose(out, expected)

def test_cuba_vector_shape_and_values():
    z = np.array([0.0, 1.5, 3.0])
    out = _sfr_cuba(z, sfr_params=None, verbose=False)
    a0, a1, a2, a3, a4, a5 = 6.9e-3, 0.14, 2.2, 1.5, 2.7, 4.1
    expected = ((a0 + a1 * (z/a2)**a3) / (1 + (z/a4)**a5))
    assert out.shape == z.shape
    assert np.allclose(out, expected)

def test_finke22a_finite_default():
    z = np.array([0.0, 1.0, 10.0])  # include values near breakpoints
    out = _sfr_finke22a(z, sfr_params=None, verbose=False)
    assert out.shape == z.shape
    assert np.all(np.isfinite(out))

def test_custom_formula_works_for_numpy_array():
    z = np.array([0.0, 1.0, 2.0], dtype=float)
    expr = "params[0] * (1+xx)**params[1]"
    params = [0.5, 2.0]
    out = sfr_model(z, expr, sfr_params=params, verbose=False)
    expected = params[0] * (1+z)**params[1]
    assert np.allclose(out, expected)

def test_custom_formula_works_for_list_xx():
    z = [0.0, 1.0, 2.0]
    expr = "params[0] * xx + params[1]"
    params = [2.0, 3.0]
    out = sfr_model(z, expr, sfr_params=params, verbose=False)
    expected = np.array(z) * 2.0 + 3.0
    assert np.allclose(out, expected)

def test_custom_formula_invalid_raises_valueerror():
    z = np.array([0.0, 1.0], dtype=float)
    expr = "params[0] * (1+xx)**does_not_exist"
    with pytest.raises(ValueError):
        sfr_model(z, expr, sfr_params=[1.0, 2.0], verbose=False)

def test_yaml_like_formula_with_params():
    z = np.array([0.0, 1.0, 2.0], dtype=float)
    expr = "params[0]*(1 + xx)**params[1]"
    params = [0.02, -0.1]
    out = sfr_model(z, expr, sfr_params=params, verbose=False)
    expected = params[0] * (1 + z) ** params[1]
    assert np.allclose(out, expected)

def test_builtin_dispatch_string():
    z = np.array([0.0, 1.0, 2.0], dtype=float)
    out = sfr_model(z, "sfr_cuba", sfr_params=None,
                    verbose=False)
    # sanity: finite output
    assert np.all(np.isfinite(out))

def test_params_none_formula_that_does_not_use_params():
    z = np.array([0.0, 1.0, 2.0], dtype=float)
    expr = "(1 + xx)**2"
    out = sfr_model(z, expr, sfr_params=None, verbose=False)
    expected = (1 + z) ** 2
    assert np.allclose(out, expected)

def test_params_none_raises_when_formula_uses_params():
    z = np.array([0.0, 1.0], dtype=float)
    expr = "params[0]*xx"
    with pytest.raises(ValueError):
        sfr_model(z, expr, sfr_params=None, verbose=False)

def test_invalid_formula_raises_valueerror():
    z = np.array([0.0, 1.0], dtype=float)
    expr = "params[0]*xx +++"  # syntax error / invalid
    with pytest.raises(ValueError):
        sfr_model(z, expr, sfr_params=[1.0], verbose=False)

def test_builtin_accepts_xx_list_for_constant():
    z_list = [0.0, 1.0, 2.0]
    out = sfr_model(z_list, "sfr_constant", sfr_params=2.0,
                    verbose=False)
    assert np.allclose(out, np.ones_like(np.array(z_list)) * 2.0)

def test_custom_formula_accepts_numpy_params_array():
    z = np.array([0.0, 1.0, 2.0], dtype=float)
    expr = "params[0]*(1 + xx)**params[1]"
    params = np.array([0.02, -0.1], dtype=float)
    out = sfr_model(z, expr, sfr_params=params, verbose=False)
    expected = params[0] * (1 + z) ** params[1]
    assert np.allclose(out, expected)

def test_custom_formula_blocks_attribute_access_security():
    z = np.array([0.0, 1.0], dtype=float)
    expr = "params[0] + xx.__class__"
    with pytest.raises(ValueError):
        sfr_model(z, expr, sfr_params=[1.0], verbose=False)
