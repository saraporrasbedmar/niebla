import numpy as np
import pytest

import src.niebla.metall_models as metall


def test_metal_constant_default():
    z = np.array([0.0, 1.0, 2.0])
    out = metall.metall_model(z, metall_model="metall_constant",
                           metall_params=None, verbose=False)

    assert isinstance(out, np.ndarray)
    assert out.shape == z.shape
    assert np.allclose(out, 0.02)


def test_metal_constant_params_float():
    z = [0.0, 1.0, 2.0]
    out = metall.metall_model(z, metall_model="metall_constant",
                              metall_params=0.1, verbose=False)
    assert np.allclose(out, np.ones(3) * 0.1)


def test_metal_tanikawa22_default_values():
    z = np.array([0.0, 1.0, 2.0])
    a, b, c, d = 0.153, 0.074, 1.34, 0.02
    expected = 10 ** (a - b * z ** c) * d

    out = metall.metall_model(z, metall_model="metall_tanikawa22",
                           metall_params=None, verbose=False)
    assert np.allclose(out, expected, rtol=1e-12, atol=0.0)


def test_metal_tanikawa22_custom_params():
    z = np.array([0.0, 1.0])
    params = [0.2, 0.05, 1.0, 0.03]
    expected = 10 ** (params[0] - params[1] * z ** params[2]) * params[3]

    out = metall.metall_model(z, metall_model="metall_tanikawa22",
                           metall_params=params, verbose=False)
    assert np.allclose(out, expected)


def test_callable_metall_params_none():
    z = np.array([0.0, 1.0, 2.0])

    def f(zz):
        return zz + 1

    out = metall.metall_model(z, metall_model=f, metall_params=None)
    assert np.allclose(out, z + 1)


def test_callable_metall_params_dict():
    z = np.array([0.0, 1.0, 2.0])

    def f(zz, scale=1.0, offset=0.0):
        return scale * zz + offset

    out = metall.metall_model(
        z,
        metall_model=f,
        metall_params={"scale": 2.0, "offset": 3.0},
    )
    assert np.allclose(out, 2.0 * z + 3.0)


def test_callable_metall_params_non_dict():
    z = np.array([0.0, 1.0, 2.0])

    def f(zz, p):
        return zz * p

    out = metall.metall_model(z, metall_model=f, metall_params=4.0)
    assert np.allclose(out, 4.0 * z)


def test_string_formula_uses_safe_formula(monkeypatch):
    z = np.array([0.0, 1.0, 2.0])

    def fake_safe_formula(formula, xx, params=None):
        # Validate inputs
        assert formula == "2*xx"
        assert np.allclose(xx, z)
        assert params == [1, 2, 3]  # passed through
        return 2 * xx

    monkeypatch.setattr(metall, "safe_formula", fake_safe_formula)

    out = metall.metall_model(
        z,
        metall_model="2*xx",
        metall_params=[1, 2, 3],
    )
    assert np.allclose(out, 2 * z)


def test_string_formula_wraps_exceptions(monkeypatch):
    z = np.array([0.0, 1.0])

    def fake_safe_formula(formula, xx, params=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(metall, "safe_formula", fake_safe_formula)

    with pytest.raises(ValueError) as excinfo:
        metall.metall_model(z, metall_model="bad_formula", metall_params=None)

    msg = str(excinfo.value)
    assert "Error evaluating string formula" in msg
    assert "bad_formula" in msg