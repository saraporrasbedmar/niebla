import numpy as np
from numpy.testing import assert_allclose
import pytest

import src.niebla.sfr_models as sfr


def test_sfr_madau14():
    zz = np.array([0., 5., 10.])
    yy_calc = sfr._sfr_madau14(zz, verbose=True)
    assert_allclose(yy_calc,
                    [0.01496149, 0.03173497, 0.00556202],
                    atol=1e-6)


def test_sfr_finke22a():
    zz = np.array([0., 5., 10.])
    yy_calc = sfr._sfr_finke22a(zz, verbose=True)
    assert_allclose(yy_calc,
                    [0.00912011, 0.02203551, 0.0015306],
                    atol=1e-6)


def test_sfr_cuba():
    zz = np.array([0., 5., 10.])
    yy_calc = sfr._sfr_cuba(zz, verbose=True)
    assert_allclose(yy_calc,
                    [0.0069, 0.03602158, 0.00632803],
                    atol=1e-6)


def test_sfr_constant():
    zz = np.array([0., 5., 10.])
    yy_calc = sfr._sfr_constant(zz, verbose=True)
    assert_allclose(yy_calc,
                    [1., 1., 1.],
                    atol=1e-6)


def test_sfr_custom():
    zz = np.array([0., 5., 10.])

    with pytest.raises(ValueError):
        sfr._sfr_custom(zz, sfr_model=0., verbose=True)

    with pytest.raises(NameError):
        sfr._sfr_custom(zz, sfr_model='z', verbose=True)
