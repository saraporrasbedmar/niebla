import numpy as np
import pytest

from src.niebla.safe_evaluation_strings import safe_formula


def test_basic_arithmetic_works():
    xx = np.array([0.0, 1.0, 2.0])
    expr = "1 + 2*xx - 0.5"
    out = safe_formula(expr, xx=xx, params=None)
    expected = 1 + 2*xx - 0.5
    assert np.allclose(out, expected)

def test_power_works():
    xx = np.array([0.0, 1.0, 2.0])
    expr = "(1 + xx)**2"
    out = safe_formula(expr, xx=xx, params=None)
    expected = (1 + xx)**2
    assert np.allclose(out, expected)

def test_unary_minus_works():
    xx = np.array([0.0, 1.0, 2.0])
    expr = "-xx + 1"
    out = safe_formula(expr, xx=xx, params=None)
    expected = -xx + 1
    assert np.allclose(out, expected)

def test_requires_x_variable_name_xx():
    xx = np.array([0.0, 1.0, 2.0])
    # Using a different name should be rejected
    with pytest.raises(ValueError):
        safe_formula("1 + z", xx=xx, params=None)

def test_params_bare_name_rejected():
    xx = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="bare 'params'"):
        safe_formula("params + 1", xx=xx, params=[1.0, 2.0])

def test_params_indexing_works_with_list_params():
    xx = np.array([0.0, 1.0, 2.0])
    expr = "params[0]*(1 + xx)**params[1]"
    params = [0.02, -0.1]
    out = safe_formula(expr, xx=xx, params=params)
    expected = params[0] * (1 + xx) ** params[1]
    assert np.allclose(out, expected)

def test_params_indexing_works_with_numpy_params():
    xx = np.array([0.0, 1.0, 2.0])
    expr = "params[1]*xx + params[0]"
    params = np.array([2.0, 3.0])
    out = safe_formula(expr, xx=xx, params=params)
    expected = params[1] * xx + params[0]
    assert np.allclose(out, expected)

def test_params_index_out_of_range_raises_valueerror():
    xx = np.array([0.0, 1.0])
    expr = "params[10] + xx"
    with pytest.raises(ValueError, match="Invalid params index"):
        safe_formula(expr, xx=xx, params=[1.0, 2.0])

def test_params_index_must_be_integer_literal():
    xx = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="params index must be an integer literal"):
        safe_formula("params[0+1] * xx", xx=xx, params=[1.0, 2.0])

def test_params_indexing_without_params_raises():
    xx = np.array([0.0, 1.0])
    expr = "params[0]*xx"
    with pytest.raises(ValueError, match="Invalid params index"):
        safe_formula(expr, xx=xx, params=None)

def test_unknown_name_rejected():
    xx = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="Unknown name"):
        safe_formula("unknown_var + xx", xx=xx, params=None)

def test_syntax_error_is_valueerror():
    xx = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="Invalid formula syntax"):
        safe_formula("xx +++", xx=xx, params=None)

def test_unsupported_expression_element_rejected():
    xx = np.array([0.0, 1.0])
    # Comparisons / boolean ops are not implemented in ev(); should raise ValueError
    with pytest.raises(ValueError):
        safe_formula("xx > 0", xx=xx, params=None)

def test_unsupported_function_calls_rejected():
    xx = np.array([0.0, 1.0])
    # Calls like sin(), log() are not implemented -> should raise ValueError
    with pytest.raises(ValueError):
        safe_formula("sin(xx)", xx=xx, params=None)

def test_attribute_access_rejected():
    xx = np.array([0.0, 1.0])
    # Attribute access isn't supported (ast.Attribute not handled)
    with pytest.raises(ValueError):
        safe_formula("xx.__class__", xx=xx, params=None)

def test_numpy_scalar_x_works():
    xx = np.float64(2.0)
    expr = "params[0] + params[1]*xx"
    params = [1.0, 3.0]
    out = safe_formula(expr, xx=xx, params=params)
    assert np.isclose(out, 1.0 + 3.0*2.0)