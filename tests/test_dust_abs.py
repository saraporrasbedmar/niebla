import numpy as np
import pytest

import src.niebla.dust_absorption_models as mod


def test_dust_abs_fraction_none_returns_ones_shape():
    wv = np.array([0.1, 0.2, 0.3])
    z = np.array([0.0, 1.0])

    out = mod.dust_abs_fraction(wv, z_array=z, models=None)

    assert out.shape == (3, 2)
    assert np.allclose(out, 1.0)


def test_dust_abs_fraction_callable_model_without_params():
    wv = np.array([0.1, 0.2])
    z = np.array([0.0, 1.0])

    def model(ww, zz):
        return np.full_like(ww, 0.25, dtype=float)

    out = mod.dust_abs_fraction(wv, z_array=z, models=model)

    assert out.shape == (2, 2)
    assert np.allclose(out, 0.25)


def test_dust_abs_fraction_callable_model_with_params():
    wv = np.array([0.1, 0.2])
    z = np.array([0.0, 1.0])

    def model(ww, zz, params):
        return np.full_like(ww, params["value"], dtype=float)

    out = mod.dust_abs_fraction(
        wv, z_array=z, models=model, dust_params={"value": 0.7}
    )

    assert out.shape == (2, 2)
    assert np.allclose(out, 0.7)


def test_dust_abs_fraction_callable_model_clips_output():
    wv = np.array([0.1, 0.2])
    z = np.array([0.0, 1.0])

    def model(ww, zz):
        return np.array([[1.5, -0.5], [0.8, 2.0]])

    out = mod.dust_abs_fraction(wv, z_array=z, models=model)

    assert out.shape == (2, 2)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
    assert np.allclose(out, np.array([[1.0, 0.0], [0.8, 1.0]]))


def test_dust_abs_fraction_string_model_expression_calls_safe_formula(monkeypatch):
    wv = np.array([0.1, 0.2])
    z = np.array([0.0, 1.0])

    called = {}

    def fake_safe_formula(expr, xx=None, params=None):
        called["expr"] = expr
        called["xx"] = xx
        called["params"] = params
        return np.zeros_like(xx[0], dtype=float)

    monkeypatch.setattr(mod, "safe_formula", fake_safe_formula)

    out = mod.dust_abs_fraction(
        wv, z_array=z, models="wv + z", dust_params={"a": 1}
    )

    assert out.shape == (2, 2)
    assert called["expr"] == "wv + z"
    assert isinstance(called["xx"], tuple)
    assert called["params"] == {"a": 1}
    assert np.allclose(out, 0.0)


def test_dust_abs_fraction_single_model_finke2022(monkeypatch):
    wv = np.array([0.1, 0.2])
    z = np.array([0.0, 1.0])

    def fake_finke2022(wv_array, z_array, dust_params, verbose=False):
        return np.zeros((len(wv_array), len(z_array)), dtype=float)

    monkeypatch.setattr(mod, "finke2022", fake_finke2022)

    out = mod.dust_abs_fraction(wv, z_array=z, models="finke2022")

    assert out.shape == (2, 2)
    assert np.allclose(out, 1.0)


def test_dust_abs_fraction_single_model_comb_model_1(monkeypatch):
    wv = np.array([0.1, 0.2])
    z = np.array([0.0, 1.0])

    def fake_comb_model_1(wv_array, z_array, dust_params, verbose=False):
        return np.zeros((len(wv_array), len(z_array)), dtype=float)

    monkeypatch.setattr(mod, "comb_model_1", fake_comb_model_1)

    out = mod.dust_abs_fraction(wv, z_array=z, models="comb_model_1")

    assert out.shape == (2, 2)
    assert np.allclose(out, 1.0)


def test_dust_abs_fraction_two_models_calls_builtin_components(monkeypatch):
    wv = np.array([0.1, 0.2])
    z = np.array([0.0, 1.0])

    monkeypatch.setattr(mod, "kneiske2002", lambda ww, dust_params, verbose=False: np.zeros_like(ww))
    monkeypatch.setattr(mod, "fermi2018", lambda zz, dust_params, verbose=False: np.zeros_like(zz))

    out = mod.dust_abs_fraction(wv, z_array=z, models=["kneiske2002", "fermi2018"])

    assert out.shape == (2, 2)
    assert np.allclose(out, 1.0)


def test_dust_abs_fraction_two_models_unknown_model_still_returns_array(monkeypatch):
    wv = np.array([0.1, 0.2])
    z = np.array([0.0, 1.0])

    monkeypatch.setattr(mod, "fermi2018", lambda zz, dust_params, verbose=False: np.zeros_like(zz))

    out = mod.dust_abs_fraction(wv, z_array=z, models=["unknown", "fermi2018"])

    assert out.shape == (2, 2)
    assert np.allclose(out, 1.0)


def test_dust_abs_fraction_invalid_models_type_raises_typeerror():
    wv = np.array([0.1, 0.2])

    with pytest.raises(TypeError):
        mod.dust_abs_fraction(wv, z_array=0.0, models=object())


def test_dust_abs_fraction_invalid_models_length_raises_valueerror():
    wv = np.array([0.1, 0.2])

    with pytest.raises(ValueError):
        mod.dust_abs_fraction(wv, z_array=0.0, models=["a", "b", "c"])


def test_kneiske2002_defaults(monkeypatch):
    wv = np.array([0.1, 1.0])

    out = mod.kneiske2002(wv, dust_params={})

    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


def test_razzaque2009_defaults(monkeypatch):
    wv = np.array([0.1, 0.2, 0.5])

    out = mod.razzaque2009(wv, dust_params={})

    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_fermi2018_defaults():
    z = np.array([0.0, 1.0, 2.0])

    out = mod.fermi2018(z, params_dust={})

    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_dust_att_finke_defaults():
    wv = np.array([0.1, 0.2, 0.5])

    out = mod.dust_att_finke(wv, params_dust={})

    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_comb_model_1_shape():
    wv = np.array([0.1, 0.2])
    z = np.array([0.0, 1.0])

    out = mod.comb_model_1(wv, z, dust_params={})

    assert out.shape == (2, 2)


def test_finke2022_shape():
    wv = np.array([0.1, 0.2])
    z = np.array([0.0, 1.0])

    out = mod.finke2022(wv, z, dust_params={})

    assert out.shape == (2, 2)


def test_nan_and_inf_cleanup_in_two_model_branch(monkeypatch):
    wv = np.array([0.1, 0.2])
    z = np.array([0.0, 1.0])

    monkeypatch.setattr(mod, "kneiske2002", lambda ww, dust_params, verbose=False: np.array([[np.nan, np.inf], [0.0, 0.0]]))
    monkeypatch.setattr(mod, "fermi2018", lambda zz, dust_params, verbose=False: np.zeros_like(zz))

    out = mod.dust_abs_fraction(wv, z_array=z, models=["kneiske2002", "fermi2018"])

    assert out.shape == (2, 2)
    assert np.isfinite(out).all()